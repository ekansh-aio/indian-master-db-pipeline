"""
Semantic Chunker Module (Enhanced with Role Awareness)
BACKWARD COMPATIBLE - Drop-in replacement for original semantic_chunker.py

Splits legal text into semantically coherent chunks using embedding similarity.
No overlap between chunks.
Includes chunk-to-document similarity calculation and top-k selection.

NEW: Optional role-awareness to prevent false chunking in legal documents.
If role_file_path is provided, uses role detection. Otherwise, works exactly as before.

Changes vs original (v2):
    FIX-G  encode_batch / _compute_document_embedding / _compute_chunk_doc_similarities:
           Removed encode_multi_process() and the multi-process SentenceTransformer
           pool entirely. encode_multi_process() spawns worker *processes* (not
           threads), each with its own CUDA context. When called from a daemon
           thread inside the pipeline those worker processes inherit a partial CUDA
           state and reliably deadlock or corrupt results. On an A100 with
           batch_size=1024, a single-process encode call already saturates the GPU;
           multi-process gives no throughput benefit and adds IPC overhead.
           The model is now moved to a single configurable device at init time and
           all encoding is done via model.encode() with large batch sizes.

    FIX-H  __init__: removed the ThreadPoolExecutor wrapper that was used to call
           encode_multi_process from a thread (unnecessary after FIX-G).

    FIX-I  _compute_document_embedding: was calling encode_multi_process directly
           (bypassing encode_batch and therefore bypassing _embed_lock in the
           pipeline). Now calls self.encode_batch() so locking is always respected.

    FIX-J  _compute_chunk_doc_similarities: same issue as FIX-I, fixed the same way.

    FIX-K  __del__: stop_multi_process_pool call removed (pool no longer exists);
           replaced with a best-effort CUDA cache clear.
"""
import re
import logging
import numpy as np
import torch
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from sentence_transformers import SentenceTransformer
from collections import Counter
import importlib.util
import sys
from config import CHUNKING_CONFIG

logger = logging.getLogger(__name__)


@dataclass
class SemanticChunk:
    """Structured semantic chunk data."""
    chunk_id: int
    text: str
    start_char: int
    end_char: int
    sentences: List[str]
    avg_similarity: float
    doc_similarity: float = 0.0
    dominant_role: Optional[str] = None
    role_distribution: Optional[Dict[str, int]] = field(default_factory=dict)
    role_purity: Optional[float] = None

    def to_dict(self) -> Dict:
        base_dict = {
            "chunk_id":      self.chunk_id,
            "text":          self.text,
            "start_char":    self.start_char,
            "end_char":      self.end_char,
            "num_sentences": len(self.sentences),
            "avg_similarity": round(self.avg_similarity, 4),
            "doc_similarity": round(self.doc_similarity, 4),
        }
        if self.dominant_role is not None:
            base_dict["dominant_role"]     = self.dominant_role
            base_dict["role_distribution"] = self.role_distribution
            base_dict["role_purity"]       = round(self.role_purity, 4) if self.role_purity else None
        return base_dict


class _LegalRoleClassifier:
    """Internal role classifier (only used if role_file_path provided)."""

    def __init__(self, role_file_path: str, model: SentenceTransformer):
        self.model = model
        self.role_file_path = role_file_path
        self.role_descriptions = {}
        self.role_embeddings = {}
        self._load_role_descriptions()
        self._compute_role_embeddings()

    def _load_role_descriptions(self):
        try:
            spec = importlib.util.spec_from_file_location("role_desc", self.role_file_path)
            if spec is None or spec.loader is None:
                raise ImportError(f"Could not load module from {self.role_file_path}")
            role_module = importlib.util.module_from_spec(spec)
            sys.modules["role_desc"] = role_module
            spec.loader.exec_module(role_module)
            self.role_descriptions = role_module.ROLE_DESCRIPTIONS_DICT
            logger.info(f"Loaded {len(self.role_descriptions)} role categories")
        except Exception as e:
            logger.error(f"Failed to load role descriptions: {e}")
            raise

    def _compute_role_embeddings(self):
        for role_name, descriptions in self.role_descriptions.items():
            combined_text = " ".join(descriptions)
            # Use model.encode directly — this is called at init time on the
            # constructor thread before the pipeline starts, so no lock needed.
            embedding = self.model.encode(combined_text, show_progress_bar=False)
            self.role_embeddings[role_name] = np.asarray(embedding)

    def classify_sentences(self, sentences: List[str], sentence_embeddings: np.ndarray) -> List[str]:
        roles = []
        for i in range(len(sentences)):
            similarities = {
                role: self._cosine_similarity(sentence_embeddings[i], emb)
                for role, emb in self.role_embeddings.items()
            }
            roles.append(max(similarities, key=similarities.get))
        return roles

    @staticmethod
    def _cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
        n1 = np.linalg.norm(emb1)
        n2 = np.linalg.norm(emb2)
        if n1 == 0 or n2 == 0:
            return 0.0
        return float(np.dot(emb1, emb2) / (n1 * n2))


class SemanticChunker:
    """
    Semantic chunker that groups sentences based on embedding similarity.

    Multi-process encoding (encode_multi_process) has been replaced with
    single-device batched encoding (FIX-G). This is required for correct
    operation inside the threaded producer/consumer pipeline: worker processes
    inherit a partial CUDA context from the spawning thread and deadlock or
    corrupt results. A single A100 with batch_size=1024 fully saturates GPU
    utilisation without any multi-process overhead.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        similarity_threshold: float = 0.5,
        min_sentences_per_chunk: int = 2,
        max_sentences_per_chunk: int = 10,
        min_chunk_size: int = 100,
        role_file_path: Optional[str] = None,
        enforce_role_boundaries: bool = True,
        role_change_penalty: float = 0.3,
    ):
        self.similarity_threshold    = similarity_threshold
        self.min_sentences_per_chunk = min_sentences_per_chunk
        self.max_sentences_per_chunk = max_sentences_per_chunk
        self.min_chunk_size          = min_chunk_size
        self.role_file_path          = role_file_path
        self.enforce_role_boundaries = enforce_role_boundaries
        self.role_change_penalty     = role_change_penalty

        # ----------------------------------------------------------------
        # FIX-G: single-device setup — no multi-process pool.
        # Pick the device BEFORE loading the model so SentenceTransformer
        # initialises its internal _target_device correctly.
        # .to(device) after the fact does NOT update that attribute, causing
        # encode() to silently run on cuda:0 regardless of what .to() was called with.
        # ----------------------------------------------------------------
        config_device = CHUNKING_CONFIG.get("device", None)
        available = torch.cuda.device_count()

        if config_device:
            self._device = torch.device(config_device)
        elif available >= 1:
            self._device = torch.device("cuda:0")
        else:
            self._device = torch.device("cpu")

        logger.info(f"Loading sentence transformer model: {model_name} on {self._device}")
        try:
            self.model = SentenceTransformer(model_name, device=str(self._device))
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

        self.model.eval()

        if available > 1:
            logger.info(
                f"SemanticChunker: {available} GPUs detected — using single device "
                f"{self._device} (multi-process pool disabled; incompatible with "
                f"threaded pipeline). Set CHUNKING_CONFIG['device'] = 'cuda:N' to "
                f"pin to a different GPU."
            )
        else:
            logger.info(f"SemanticChunker: using device {self._device}")

        # FIX-H: no ThreadPoolExecutor needed (pool removed)
        self._encode_batch_size = CHUNKING_CONFIG.get("encode_batch_size", 1024)

        # Pre-compile patterns
        self.sentence_pattern   = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')
        self.whitespace_pattern = re.compile(r'\s+')

        # Role classifier (optional)
        self.role_classifier = None
        if role_file_path:
            logger.info(f"Role-awareness enabled with file: {role_file_path}")
            self.role_classifier = _LegalRoleClassifier(role_file_path, self.model)
        else:
            logger.info("Role-awareness disabled (backward compatible mode)")

    def __del__(self):
        # FIX-K: pool is gone; best-effort CUDA cache clear instead
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # BATCH-PIPELINE METHODS (used by producer/consumer pipeline)
    # ------------------------------------------------------------------

    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """
        GPU: encode a flat list of texts in one call. Returns float32 array.

        FIX-G: encode_multi_process() removed. All encoding uses model.encode()
        on the single pre-configured device. The caller (production_pipeline.py)
        wraps this in _embed_lock so only one thread enters at a time.
        """
        if not texts:
            # Return correct embedding dimension even for empty input
            dim = self.model.get_sentence_embedding_dimension() or 384
            return np.empty((0, dim), dtype=np.float32)

        result = self.model.encode(
            texts,
            batch_size=self._encode_batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return np.asarray(result, dtype=np.float32)

    def assemble_chunks_cpu(
        self,
        text: str,
        sentences: List[str],
        sentence_embeddings: np.ndarray,
    ) -> Tuple[List["SemanticChunk"], List[str]]:
        """
        CPU-only: build SemanticChunk objects from pre-computed embeddings.

        Returns (chunks, chunk_texts). doc_similarity is left at 0.0 — caller
        fills it in after encoding chunk_texts via encode_batch().
        """
        roles = None
        if self.role_classifier:
            roles = self.role_classifier.classify_sentences(sentences, sentence_embeddings)

        chunk_groups = self._create_semantic_chunks(sentences, sentence_embeddings, roles)

        chunks_out: List[SemanticChunk] = []
        chunk_texts: List[str] = []
        current_pos = 0
        skipped = 0

        for cid, sent_indices in enumerate(chunk_groups, start=1):
            chunk_sents = [sentences[i] for i in sent_indices]
            chunk_text  = " ".join(chunk_sents)
            if len(chunk_text) < self.min_chunk_size:
                skipped += 1
                continue
            chunk_texts.append(chunk_text)

            if len(sent_indices) > 1:
                c_embs = sentence_embeddings[sent_indices]
                sims   = [self._compute_similarity(c_embs[i], c_embs[i + 1])
                          for i in range(len(c_embs) - 1)]
                avg_sim = float(np.mean(sims))
            else:
                avg_sim = 1.0

            dominant_role = role_dist = role_purity = None
            if roles:
                cr          = [roles[i] for i in sent_indices]
                role_dist   = dict(Counter(cr))
                dominant_role = max(role_dist, key=role_dist.get)
                role_purity = role_dist[dominant_role] / len(cr)

            sc = text.find(chunk_text, current_pos)
            if sc == -1:
                sc = text.find(chunk_sents[0], current_pos)
                if sc == -1:
                    sc = current_pos
            ec = sc + len(chunk_text)

            chunks_out.append(SemanticChunk(
                chunk_id=cid - skipped,
                text=chunk_text,
                start_char=sc,
                end_char=ec,
                sentences=chunk_sents,
                avg_similarity=avg_sim,
                doc_similarity=0.0,
                dominant_role=dominant_role,
                role_distribution=role_dist,
                role_purity=role_purity,
            ))
            current_pos = ec

        return chunks_out, chunk_texts

    # ------------------------------------------------------------------
    # INTERNAL HELPERS
    # ------------------------------------------------------------------

    def _split_sentences(self, text: str) -> List[str]:
        text      = self.whitespace_pattern.sub(" ", text).strip()
        sentences = self.sentence_pattern.split(text)
        return [s.strip() for s in sentences if s.strip()]

    def _compute_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        n1 = np.linalg.norm(emb1)
        n2 = np.linalg.norm(emb2)
        if n1 == 0 or n2 == 0:
            return 0.0
        return float(np.dot(emb1, emb2) / (n1 * n2))

    def _create_semantic_chunks(
        self,
        sentences: List[str],
        embeddings: np.ndarray,
        roles: Optional[List[str]] = None,
    ) -> List[List[int]]:
        if len(sentences) == 0:
            return []
        if len(sentences) == 1:
            return [[0]]

        chunks       = []
        current_chunk = [0]
        current_role  = roles[0] if roles else None
        similarities  = []

        for i in range(1, len(sentences)):
            similarity    = self._compute_similarity(embeddings[i - 1], embeddings[i])
            similarities.append(similarity)

            role_changed  = False
            effective_sim = similarity

            if roles and self.enforce_role_boundaries:
                role_changed = roles[i] != current_role
                if role_changed:
                    effective_sim = similarity - self.role_change_penalty
                    logger.debug(
                        f"Role change at sentence {i}: {current_role} → {roles[i]}. "
                        f"Similarity: {similarity:.3f} → {effective_sim:.3f}"
                    )

            should_split = (
                effective_sim < self.similarity_threshold
                or len(current_chunk) >= self.max_sentences_per_chunk
                or (role_changed and self.enforce_role_boundaries)
            )

            if should_split and len(current_chunk) >= self.min_sentences_per_chunk:
                chunks.append(current_chunk)
                current_chunk = [i]
                if roles:
                    current_role = roles[i]
            else:
                current_chunk.append(i)

        if current_chunk:
            if len(current_chunk) < self.min_sentences_per_chunk and chunks:
                chunks[-1].extend(current_chunk)
            else:
                chunks.append(current_chunk)

        logger.debug(
            f"Created {len(chunks)} semantic chunks from {len(sentences)} sentences. "
            f"Avg similarity: {np.mean(similarities):.3f}"
        )
        return chunks

    def _compute_document_embedding(self, text: str) -> np.ndarray:
        """
        FIX-I: was calling encode_multi_process directly, bypassing _embed_lock.
        Now delegates to encode_batch() so the lock is always respected.
        """
        logger.debug("Computing document embedding")
        return self.encode_batch([text])[0]

    def _compute_chunk_doc_similarities(
        self,
        chunk_texts: List[str],
        doc_embedding: np.ndarray,
    ) -> List[float]:
        """
        FIX-J: was calling encode_multi_process directly, bypassing _embed_lock.
        Now delegates to encode_batch() so the lock is always respected.
        """
        logger.debug(f"Computing chunk-to-document similarities for {len(chunk_texts)} chunks")
        chunk_embeddings = self.encode_batch(chunk_texts)
        return [self._compute_similarity(emb, doc_embedding) for emb in chunk_embeddings]

    # ------------------------------------------------------------------
    # PUBLIC split() API  (backward compatible)
    # ------------------------------------------------------------------

    def split(
        self,
        text: str,
        compute_doc_similarity: bool = True,
    ) -> Tuple[List[Dict], Optional[np.ndarray]]:
        """Split text into semantic chunks (backward compatible public API)."""
        if not text:
            logger.warning("Empty text provided for splitting")
            return [], None

        mode = "role-aware" if self.role_classifier else "semantic-only"
        logger.info(f"Starting {mode} chunking. Text length: {len(text)} chars")

        try:
            doc_embedding = None
            if compute_doc_similarity:
                doc_embedding = self._compute_document_embedding(text)

            sentences = self._split_sentences(text)
            if not sentences:
                logger.warning("No sentences extracted from text")
                return [], doc_embedding

            logger.info(f"Extracted {len(sentences)} sentences")

            embeddings = self.encode_batch(sentences)
            logger.debug(f"Generated embeddings with shape {embeddings.shape}")

            roles = None
            if self.role_classifier:
                roles = self.role_classifier.classify_sentences(sentences, embeddings)
                logger.info(f"Role distribution: {dict(Counter(roles))}")

            chunk_groups = self._create_semantic_chunks(sentences, embeddings, roles)

            chunks: List[SemanticChunk] = []
            chunk_texts: List[str] = []
            current_pos  = 0
            skipped_count = 0

            for chunk_id, sentence_indices in enumerate(chunk_groups, start=1):
                chunk_sentences = [sentences[i] for i in sentence_indices]
                chunk_text      = " ".join(chunk_sentences)

                if len(chunk_text) < self.min_chunk_size:
                    skipped_count += 1
                    continue

                chunk_texts.append(chunk_text)

                if len(sentence_indices) > 1:
                    c_embs = embeddings[sentence_indices]
                    sims   = [self._compute_similarity(c_embs[i], c_embs[i + 1])
                              for i in range(len(c_embs) - 1)]
                    avg_similarity = float(np.mean(sims))
                else:
                    avg_similarity = 1.0

                dominant_role = role_dist = role_purity = None
                if roles:
                    chunk_roles   = [roles[i] for i in sentence_indices]
                    role_dist     = dict(Counter(chunk_roles))
                    dominant_role = max(role_dist, key=role_dist.get)
                    role_purity   = role_dist[dominant_role] / len(chunk_roles)

                start_char = text.find(chunk_text, current_pos)
                if start_char == -1:
                    start_char = text.find(chunk_sentences[0], current_pos)
                    if start_char == -1:
                        start_char = current_pos
                end_char = start_char + len(chunk_text)

                chunks.append(SemanticChunk(
                    chunk_id=chunk_id - skipped_count,
                    text=chunk_text,
                    start_char=start_char,
                    end_char=end_char,
                    sentences=chunk_sentences,
                    avg_similarity=avg_similarity,
                    doc_similarity=0.0,
                    dominant_role=dominant_role,
                    role_distribution=role_dist,
                    role_purity=role_purity,
                ))
                current_pos = end_char

            if compute_doc_similarity and doc_embedding is not None and chunk_texts:
                doc_sims = self._compute_chunk_doc_similarities(chunk_texts, doc_embedding)
                for chunk, doc_sim in zip(chunks, doc_sims):
                    chunk.doc_similarity = doc_sim
                logger.info(
                    f"Chunk-to-doc similarities — "
                    f"avg: {np.mean(doc_sims):.4f}, "
                    f"min: {np.min(doc_sims):.4f}, "
                    f"max: {np.max(doc_sims):.4f}"
                )

            logger.info(
                f"Chunking complete. Created {len(chunks)} chunks, "
                f"skipped {skipped_count} small chunks."
            )
            return [c.to_dict() for c in chunks], doc_embedding

        except Exception as e:
            logger.error(f"Error during semantic chunking: {e}", exc_info=True)
            raise

    def select_top_k_chunks(
        self,
        chunks: List[Dict],
        k: int,
        sort_by: str = "doc_similarity",
    ) -> List[Dict]:
        if not chunks:
            return []
        if k <= 0:
            return []
        if k >= len(chunks):
            return chunks
        sorted_chunks  = sorted(chunks, key=lambda x: x.get(sort_by, 0), reverse=True)
        top_k          = sorted_chunks[:k]
        return sorted(top_k, key=lambda x: x.get("chunk_id", 0))


# ------------------------------------------------------------------
# BACKWARD COMPATIBLE CONVENIENCE FUNCTIONS
# ------------------------------------------------------------------

def split_into_semantic_chunks(
    text: str,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    similarity_threshold: float = 0.8,
    min_sentences_per_chunk: int = 3,
    max_sentences_per_chunk: int = 11,
    compute_doc_similarity: bool = True,
    role_file_path: Optional[str] = None,
    enforce_role_boundaries: bool = True,
    role_change_penalty: float = 0.3,
) -> Tuple[List[Dict], Optional[np.ndarray]]:
    chunker = SemanticChunker(
        model_name=model_name,
        similarity_threshold=similarity_threshold,
        min_sentences_per_chunk=min_sentences_per_chunk,
        max_sentences_per_chunk=max_sentences_per_chunk,
        role_file_path=role_file_path,
        enforce_role_boundaries=enforce_role_boundaries,
        role_change_penalty=role_change_penalty,
    )
    return chunker.split(text, compute_doc_similarity=compute_doc_similarity)


def select_top_k_chunks_from_text(
    text: str,
    k: int,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    similarity_threshold: float = 0.8,
    min_sentences_per_chunk: int = 3,
    max_sentences_per_chunk: int = 11,
    sort_by: str = "doc_similarity",
    role_file_path: Optional[str] = None,
    enforce_role_boundaries: bool = True,
    role_change_penalty: float = 0.3,
) -> List[Dict]:
    chunker = SemanticChunker(
        model_name=model_name,
        similarity_threshold=similarity_threshold,
        min_sentences_per_chunk=min_sentences_per_chunk,
        max_sentences_per_chunk=max_sentences_per_chunk,
        role_file_path=role_file_path,
        enforce_role_boundaries=enforce_role_boundaries,
        role_change_penalty=role_change_penalty,
    )
    chunks, _ = chunker.split(text, compute_doc_similarity=True)
    return chunker.select_top_k_chunks(chunks, k, sort_by=sort_by)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    sample = """
    The appellant filed an appeal to the Supreme Court challenging the lower court's decision.
    The case involved a dispute over contract interpretation and breach of terms.
    The respondent argued that all contractual obligations were fulfilled in good faith.

    The court examined the evidence presented by both parties in detail.
    Witness testimonies were considered along with documentary evidence.
    The contract clauses were analyzed in the context of applicable law.

    After careful consideration, the court found merit in the appellant's arguments.
    The judgment of the lower court was set aside.
    The matter was remanded for fresh consideration with specific directions.

    The Supreme Court emphasized the importance of interpreting contracts fairly.
    Both parties were directed to bear their own costs.
    The decision was delivered unanimously by a three-judge bench.
    """

    chunks, doc_emb = split_into_semantic_chunks(
        sample,
        similarity_threshold=0.5,
        min_sentences_per_chunk=2,
        max_sentences_per_chunk=5,
        compute_doc_similarity=True,
    )

    print(f"\nTotal semantic chunks: {len(chunks)}")
    for chunk in chunks:
        print(f"\nChunk {chunk['chunk_id']}:")
        print(f"  Sentences:    {chunk['num_sentences']}")
        print(f"  Avg Sim:      {chunk['avg_similarity']}")
        print(f"  Doc Sim:      {chunk['doc_similarity']}")
        print(f"  Text:         {chunk['text'][:100]}...")
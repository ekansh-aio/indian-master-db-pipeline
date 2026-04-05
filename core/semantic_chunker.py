"""
Semantic Chunker Module (Enhanced with Role Awareness)
BACKWARD COMPATIBLE - Drop-in replacement for original semantic_chunker.py

Splits legal text into semantically coherent chunks using embedding similarity.
No overlap between chunks.
Includes chunk-to-document similarity calculation and top-k selection.

NEW: Optional role-awareness to prevent false chunking in legal documents.
If role_file_path is provided, uses role detection. Otherwise, works exactly as before.
"""
import re
import logging
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from sentence_transformers import SentenceTransformer
from collections import Counter
import importlib.util
import sys

# Configure logger
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
    # New optional fields for role-awareness (backward compatible)
    dominant_role: Optional[str] = None
    role_distribution: Optional[Dict[str, int]] = field(default_factory=dict)
    role_purity: Optional[float] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary format."""
        base_dict = {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "num_sentences": len(self.sentences),
            "avg_similarity": round(self.avg_similarity, 4),
            "doc_similarity": round(self.doc_similarity, 4)
        }
        
        # Add role fields only if they exist (backward compatible)
        if self.dominant_role is not None:
            base_dict["dominant_role"] = self.dominant_role
            base_dict["role_distribution"] = self.role_distribution
            base_dict["role_purity"] = round(self.role_purity, 4) if self.role_purity else None
        
        return base_dict


class _LegalRoleClassifier:
    """
    Internal role classifier (only used if role_file_path provided).
    Not exposed in public API.
    """
    
    def __init__(self, role_file_path: str, model: SentenceTransformer):
        self.model = model
        self.role_file_path = role_file_path
        self.role_descriptions = {}
        self.role_embeddings = {}
        
        self._load_role_descriptions()
        self._compute_role_embeddings()
    
    def _load_role_descriptions(self):
        """Load role descriptions from external Python file."""
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
        """Compute embeddings for each role's description texts."""
        for role_name, descriptions in self.role_descriptions.items():
            combined_text = " ".join(descriptions)
            embedding = self.model.encode(combined_text, show_progress_bar=False)
            self.role_embeddings[role_name] = np.asarray(embedding)
    
    def classify_sentences(self, sentences: List[str], sentence_embeddings: np.ndarray) -> List[str]:
        """Classify multiple sentences efficiently."""
        roles = []
        for i, sentence in enumerate(sentences):
            # Compute similarity with each role
            similarities = {}
            for role_name, role_embedding in self.role_embeddings.items():
                sim = self._cosine_similarity(sentence_embeddings[i], role_embedding)
                similarities[role_name] = sim
            
            # Return role with highest similarity
            best_role = max(similarities.items(), key=lambda x: x[1])[0]
            roles.append(best_role)
        
        return roles
    
    @staticmethod
    def _cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings."""
        dot_product = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)


class SemanticChunker:
    """
    Semantic chunker that groups sentences based on embedding similarity.
    Uses cosine similarity between consecutive sentences to determine chunk boundaries.
    No overlap between chunks.
    Includes chunk-to-document similarity calculation.
    
    NEW: Optional role-awareness to prevent false chunking.
    - If role_file_path is provided: Uses role-aware chunking
    - If role_file_path is None: Works exactly as original (backward compatible)
    """
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        similarity_threshold: float = 0.5,
        min_sentences_per_chunk: int = 2,
        max_sentences_per_chunk: int = 10,
        min_chunk_size: int = 100,
        # NEW OPTIONAL PARAMETERS (backward compatible - default to None)
        role_file_path: Optional[str] = None,
        enforce_role_boundaries: bool = True,
        role_change_penalty: float = 0.3
    ):
        """
        Initialize semantic chunker.
        
        Args:
            model_name: HuggingFace model for sentence embeddings
            similarity_threshold: Cosine similarity threshold for grouping (0-1)
            min_sentences_per_chunk: Minimum sentences per chunk
            max_sentences_per_chunk: Maximum sentences per chunk
            min_chunk_size: Minimum characters for valid chunks
            role_file_path: Optional path to role descriptions file (enables role-awareness)
            enforce_role_boundaries: If True, force split on role changes (only if role_file_path provided)
            role_change_penalty: Similarity penalty when role changes (only if role_file_path provided)
        """
        self.similarity_threshold = similarity_threshold
        self.min_sentences_per_chunk = min_sentences_per_chunk
        self.max_sentences_per_chunk = max_sentences_per_chunk
        self.min_chunk_size = min_chunk_size
        self.role_file_path = role_file_path
        self.enforce_role_boundaries = enforce_role_boundaries
        self.role_change_penalty = role_change_penalty
        
        logger.info(f"Loading sentence transformer model: {model_name}")
        try:
            self.model = SentenceTransformer(model_name)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
        
        # Initialize role classifier only if role_file_path provided
        self.role_classifier = None
        if role_file_path:
            logger.info(f"Role-awareness enabled with file: {role_file_path}")
            self.role_classifier = _LegalRoleClassifier(role_file_path, self.model)
        else:
            logger.info("Role-awareness disabled (backward compatible mode)")
        
        # Pre-compile sentence splitting pattern
        self.sentence_pattern = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')
        self.whitespace_pattern = re.compile(r'\s+')
        
        logger.info(
            f"SemanticChunker initialized with threshold={similarity_threshold}, "
            f"min_sentences={min_sentences_per_chunk}, max_sentences={max_sentences_per_chunk}"
        )
    
    def _split_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences.
        
        Args:
            text: Input text
            
        Returns:
            List of sentences
        """
        # Normalize whitespace
        text = self.whitespace_pattern.sub(' ', text).strip()
        
        # Split on sentence boundaries
        sentences = self.sentence_pattern.split(text)
        
        # Clean and filter
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences
    
    def _compute_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            emb1: First embedding vector
            emb2: Second embedding vector
            
        Returns:
            Cosine similarity score
        """
        dot_product = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def _create_semantic_chunks(
        self,
        sentences: List[str],
        embeddings: np.ndarray,
        roles: Optional[List[str]] = None
    ) -> List[List[int]]:
        """
        Group sentences into chunks based on semantic similarity.
        Optionally considers role changes if role classifier is enabled.
        
        Args:
            sentences: List of sentence strings
            embeddings: Sentence embeddings matrix
            roles: Optional list of role classifications (if role-aware mode)
            
        Returns:
            List of sentence index groups
        """
        if len(sentences) == 0:
            return []
        
        if len(sentences) == 1:
            return [[0]]
        
        chunks = []
        current_chunk = [0]
        current_role = roles[0] if roles else None
        similarities = []
        
        for i in range(1, len(sentences)):
            # Compute similarity with previous sentence
            similarity = self._compute_similarity(embeddings[i-1], embeddings[i])
            similarities.append(similarity)
            
            # Check for role change (if role-aware mode)
            role_changed = False
            effective_sim = similarity
            
            if roles and self.enforce_role_boundaries:
                role_changed = roles[i] != current_role
                if role_changed:
                    # Apply penalty to similarity when role changes
                    effective_sim = similarity - self.role_change_penalty
                    logger.debug(
                        f"Role change at sentence {i}: {current_role} -> {roles[i]}. "
                        f"Similarity: {similarity:.3f} -> {effective_sim:.3f}"
                    )
            
            # Check if we should start a new chunk
            should_split = (
                effective_sim < self.similarity_threshold or
                len(current_chunk) >= self.max_sentences_per_chunk or
                (role_changed and self.enforce_role_boundaries)
            )
            
            if should_split and len(current_chunk) >= self.min_sentences_per_chunk:
                chunks.append(current_chunk)
                current_chunk = [i]
                if roles:
                    current_role = roles[i]
            else:
                current_chunk.append(i)
        
        # Add final chunk
        if current_chunk:
            # If last chunk is too small, merge with previous
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
        Compute embedding for the entire document.
        
        Args:
            text: Full document text
            
        Returns:
            Document embedding vector
        """
        logger.debug("Computing document embedding")
        doc_embedding = self.model.encode(text, show_progress_bar=False)
        return np.asarray(doc_embedding)
    
    def _compute_chunk_doc_similarities(
        self,
        chunk_texts: List[str],
        doc_embedding: np.ndarray
    ) -> List[float]:
        """
        Compute similarity between each chunk and the document.
        
        Args:
            chunk_texts: List of chunk text strings
            doc_embedding: Document embedding vector
            
        Returns:
            List of similarity scores
        """
        logger.debug(f"Computing chunk-to-document similarities for {len(chunk_texts)} chunks")
        
        # Generate chunk embeddings
        chunk_embeddings = self.model.encode(chunk_texts, show_progress_bar=False)
        chunk_embeddings = np.asarray(chunk_embeddings)
        
        # Compute similarities
        similarities = []
        for chunk_emb in chunk_embeddings:
            sim = self._compute_similarity(chunk_emb, doc_embedding)
            similarities.append(sim)
        
        return similarities
    
    def split(self, text: str, compute_doc_similarity: bool = True) -> Tuple[List[Dict], Optional[np.ndarray]]:
        """
        Split text into semantic chunks with character offsets.
        No overlap between chunks.
        Optionally compute chunk-to-document similarity.
        
        If role_file_path was provided during init, uses role-aware chunking.
        Otherwise, uses pure semantic chunking (backward compatible).
        
        Args:
            text: Input text to split
            compute_doc_similarity: Whether to compute chunk-to-document similarity
            
        Returns:
            Tuple of (list of chunk dictionaries with metadata, document embedding)
        """
        if not text:
            logger.warning("Empty text provided for splitting")
            return [], None
        
        original_length = len(text)
        mode = "role-aware" if self.role_classifier else "semantic-only"
        logger.info(f"Starting {mode} chunking. Text length: {original_length} chars")
        
        try:
            # Compute document embedding first if needed
            doc_embedding = None
            if compute_doc_similarity:
                doc_embedding = self._compute_document_embedding(text)
            
            # Step 1: Split into sentences
            logger.debug("Splitting text into sentences")
            sentences = self._split_sentences(text)
            
            if not sentences:
                logger.warning("No sentences extracted from text")
                return [], doc_embedding
            
            logger.info(f"Extracted {len(sentences)} sentences")
            
            # Step 2: Generate embeddings
            logger.debug("Generating sentence embeddings")
            embeddings = self.model.encode(sentences, show_progress_bar=False)
            embeddings = np.asarray(embeddings)
            logger.debug(f"Generated embeddings with shape {embeddings.shape}")
            
            # Step 3: Classify sentences into roles (if role-aware mode)
            roles = None
            if self.role_classifier:
                logger.debug("Classifying sentences into legal roles")
                roles = self.role_classifier.classify_sentences(sentences, embeddings)
                role_counter = Counter(roles)
                logger.info(f"Role distribution: {dict(role_counter)}")
            
            # Step 4: Create semantic chunks (with or without role awareness)
            logger.debug("Creating semantic chunks")
            chunk_groups = self._create_semantic_chunks(sentences, embeddings, roles)
            
            # Step 5: Build chunk objects with offsets
            chunks: List[SemanticChunk] = []
            current_pos = 0
            skipped_count = 0
            
            # First pass: create chunks
            chunk_texts = []
            for chunk_id, sentence_indices in enumerate(chunk_groups, start=1):
                # Combine sentences in this chunk
                chunk_sentences = [sentences[i] for i in sentence_indices]
                chunk_text = ' '.join(chunk_sentences)
                
                # Skip if chunk is too small
                if len(chunk_text) < self.min_chunk_size:
                    skipped_count += 1
                    continue
                
                chunk_texts.append(chunk_text)
                
                # Calculate average similarity within chunk
                if len(sentence_indices) > 1:
                    chunk_embeddings = embeddings[sentence_indices]
                    similarities = []
                    for i in range(len(chunk_embeddings) - 1):
                        sim = self._compute_similarity(
                            chunk_embeddings[i],
                            chunk_embeddings[i + 1]
                        )
                        similarities.append(sim)
                    avg_similarity = np.mean(similarities)
                else:
                    avg_similarity = 1.0
                
                # Compute role metadata (if role-aware mode)
                dominant_role = None
                role_dist = None
                role_purity = None
                
                if roles:
                    chunk_roles = [roles[i] for i in sentence_indices]
                    role_dist = dict(Counter(chunk_roles))
                    dominant_role = max(role_dist.items(), key=lambda x: x[1])[0]
                    role_purity = role_dist[dominant_role] / len(chunk_roles)
                
                # Find chunk position in original text
                start_char = text.find(chunk_text, current_pos)
                if start_char == -1:
                    # Fallback: try finding first sentence
                    first_sentence = chunk_sentences[0]
                    start_char = text.find(first_sentence, current_pos)
                    if start_char == -1:
                        start_char = current_pos
                
                end_char = start_char + len(chunk_text)
                
                chunks.append(
                    SemanticChunk(
                        chunk_id=chunk_id - skipped_count,
                        text=chunk_text,
                        start_char=start_char,
                        end_char=end_char,
                        sentences=chunk_sentences,
                        avg_similarity=float(avg_similarity),
                        doc_similarity=0.0,  # Will be updated below
                        dominant_role=dominant_role,
                        role_distribution=role_dist,
                        role_purity=role_purity
                    )
                )
                
                current_pos = end_char
            
            # Step 6: Compute chunk-to-document similarities
            if compute_doc_similarity and doc_embedding is not None and chunk_texts:
                doc_similarities = self._compute_chunk_doc_similarities(chunk_texts, doc_embedding)
                
                # Update chunks with document similarities
                for chunk, doc_sim in zip(chunks, doc_similarities):
                    chunk.doc_similarity = doc_sim
                
                logger.info(
                    f"Computed chunk-to-doc similarities. "
                    f"Avg: {np.mean(doc_similarities):.4f}, "
                    f"Min: {np.min(doc_similarities):.4f}, "
                    f"Max: {np.max(doc_similarities):.4f}"
                )
            
            # Log final statistics
            if roles:
                logger.info(
                    f"Role-aware chunking complete. "
                    f"Created {len(chunks)} chunks, skipped {skipped_count} small chunks. "
                )
            else:
                logger.info(
                    f"Semantic chunking complete. "
                    f"Created {len(chunks)} chunks, skipped {skipped_count} small chunks"
                )
            
            # Convert to dict format
            return [c.to_dict() for c in chunks], doc_embedding
            
        except Exception as e:
            logger.error(f"Error during semantic chunking: {e}", exc_info=True)
            raise
    
    def select_top_k_chunks(
        self,
        chunks: List[Dict],
        k: int,
        sort_by: str = "doc_similarity"
    ) -> List[Dict]:
        """
        Select top k chunks based on a scoring criterion.
        
        Args:
            chunks: List of chunk dictionaries
            k: Number of chunks to select
            sort_by: Field to sort by ('doc_similarity', 'avg_similarity', or 'role_purity')
            
        Returns:
            List of top k chunks, sorted by original chunk order
        """
        if not chunks:
            logger.warning("No chunks provided for top-k selection")
            return []
        
        if k <= 0:
            logger.warning(f"Invalid k value: {k}. Returning empty list.")
            return []
        
        if k >= len(chunks):
            logger.info(f"k ({k}) >= number of chunks ({len(chunks)}). Returning all chunks.")
            return chunks
        
        logger.info(f"Selecting top {k} chunks from {len(chunks)} total chunks based on {sort_by}")
        
        # Sort by the specified criterion
        sorted_chunks = sorted(chunks, key=lambda x: x.get(sort_by, 0), reverse=True)
        
        # Select top k
        top_k = sorted_chunks[:k]
        
        # Re-sort by original chunk_id to maintain document order
        top_k_sorted = sorted(top_k, key=lambda x: x.get("chunk_id", 0))
        
        logger.info(
            f"Selected top {k} chunks. "
            f"Score range: [{top_k_sorted[0].get(sort_by, 0):.4f} - {top_k_sorted[-1].get(sort_by, 0):.4f}]"
        )
        
        return top_k_sorted


# BACKWARD COMPATIBLE CONVENIENCE FUNCTIONS
# These maintain the exact same signatures as the original

def split_into_semantic_chunks(
    text: str,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    similarity_threshold: float = 0.8,
    min_sentences_per_chunk: int = 3,
    max_sentences_per_chunk: int = 11,
    compute_doc_similarity: bool = True,
    # NEW OPTIONAL PARAMETERS (backward compatible)
    role_file_path: Optional[str] = None,
    enforce_role_boundaries: bool = True,
    role_change_penalty: float = 0.3
) -> Tuple[List[Dict], Optional[np.ndarray]]:
    """
    Split text into semantic chunks with no overlap.
    
    BACKWARD COMPATIBLE: Works exactly as before if role_file_path is not provided.
    NEW: Optionally enable role-awareness by providing role_file_path.
    
    Args:
        text: Input text
        model_name: Sentence transformer model name
        similarity_threshold: Similarity threshold for grouping
        min_sentences_per_chunk: Minimum sentences per chunk
        max_sentences_per_chunk: Maximum sentences per chunk
        compute_doc_similarity: Whether to compute chunk-to-document similarity
        role_file_path: Optional path to role descriptions file (enables role-awareness)
        enforce_role_boundaries: Force split on role changes (only if role_file_path provided)
        role_change_penalty: Similarity penalty for role changes (only if role_file_path provided)
        
    Returns:
        Tuple of (list of chunk dictionaries, document embedding)
    """
    chunker = SemanticChunker(
        model_name=model_name,
        similarity_threshold=similarity_threshold,
        min_sentences_per_chunk=min_sentences_per_chunk,
        max_sentences_per_chunk=max_sentences_per_chunk,
        role_file_path=role_file_path,
        enforce_role_boundaries=enforce_role_boundaries,
        role_change_penalty=role_change_penalty
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
    # NEW OPTIONAL PARAMETERS (backward compatible)
    role_file_path: Optional[str] = None,
    enforce_role_boundaries: bool = True,
    role_change_penalty: float = 0.3
) -> List[Dict]:
    """
    Convenience function to chunk text and select top k chunks in one call.
    
    BACKWARD COMPATIBLE: Works exactly as before if role_file_path is not provided.
    NEW: Optionally enable role-awareness by providing role_file_path.
    
    Args:
        text: Input text
        k: Number of chunks to select
        model_name: Sentence transformer model name
        similarity_threshold: Similarity threshold for grouping
        min_sentences_per_chunk: Minimum sentences per chunk
        max_sentences_per_chunk: Maximum sentences per chunk
        sort_by: Field to sort by ('doc_similarity', 'avg_similarity', or 'role_purity')
        role_file_path: Optional path to role descriptions file (enables role-awareness)
        enforce_role_boundaries: Force split on role changes (only if role_file_path provided)
        role_change_penalty: Similarity penalty for role changes (only if role_file_path provided)
        
    Returns:
        List of top k chunks
    """
    chunker = SemanticChunker(
        model_name=model_name,
        similarity_threshold=similarity_threshold,
        min_sentences_per_chunk=min_sentences_per_chunk,
        max_sentences_per_chunk=max_sentences_per_chunk,
        role_file_path=role_file_path,
        enforce_role_boundaries=enforce_role_boundaries,
        role_change_penalty=role_change_penalty
    )
    
    chunks, _ = chunker.split(text, compute_doc_similarity=True)
    
    return chunker.select_top_k_chunks(chunks, k, sort_by=sort_by)


if __name__ == "__main__":
    # Configure logging for standalone execution
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Test with sample text
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
    
    print("\n=== BACKWARD COMPATIBLE MODE (SEMANTIC-ONLY) ===")
    # Works exactly as before - no role_file_path provided
    chunks, doc_emb = split_into_semantic_chunks(
        sample,
        similarity_threshold=0.5,
        min_sentences_per_chunk=2,
        max_sentences_per_chunk=5,
        compute_doc_similarity=True
        # role_file_path NOT provided = backward compatible mode
    )
    
    print(f"\nTotal semantic chunks: {len(chunks)}")
    for chunk in chunks:
        print(f"\nChunk {chunk['chunk_id']}:")
        print(f"  Sentences: {chunk['num_sentences']}")
        print(f"  Avg Similarity: {chunk['avg_similarity']}")
        print(f"  Doc Similarity: {chunk['doc_similarity']}")
        print(f"  Has role info: {'dominant_role' in chunk}")
        print(f"  Text: {chunk['text'][:100]}...")
    
    print("\n\n=== ENHANCED MODE (ROLE-AWARE) ===")
    print("Note: Requires role_desc.py file to run this mode")
    print("Usage:")
    print("""
    chunks, doc_emb = split_into_semantic_chunks(
        sample,
        similarity_threshold=0.5,
        min_sentences_per_chunk=2,
        max_sentences_per_chunk=5,
        role_file_path="/path/to/role_desc.py",  # Enable role-awareness
        enforce_role_boundaries=True,
        role_change_penalty=0.3
    )
    """)
"""
Test Pipeline for Legal Document Processing
- Fine-tuned role classification
- Weighted proportional top-k selection
- same_role_chunk_ids attached to every chunk
- Saves all chunks locally in mirrored directory structure
"""
import json
import logging
from utils.json_helper import safe_json_dump
import time
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import hashlib
import sys

from utils.weighted_selector import weighted_topk_selection

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Import configuration
import argparse

from config import (
    LOGGING_CONFIG,
    ADLS_CONFIG,
    EMBEDDING_CONFIG,
    CHUNKING_CONFIG,
    PROCESSING_CONFIG,
    ROLE_CLASSIFICATION_CONFIG,
    DOC_TYPE_CONFIG,
    INDEX_TYPE_CONFIG,
    PIPELINE_CONFIG
)

# Import pipeline components
from core.adls_fetcher import ADLSFetcher
from core.legal_text_cleaner import LegalTextCleaner
from core.semantic_chunker import SemanticChunker
from tqdm import tqdm
from core.role_classifier import create_classifier_from_config

logger = logging.getLogger(__name__)


def generate_document_id_from_path(source_file_path: str) -> str:
    path_without_ext = source_file_path.replace('.json', '')
    parts = Path(path_without_ext).parts
    skip_prefixes = {'raw', 'newapp', 'input', 'data'}
    relevant_parts = [p for p in parts if p.lower() not in skip_prefixes]
    doc_id = '_'.join(relevant_parts).replace('-', '_').replace(' ', '_').lower()
    if len(doc_id) > 200:
        base = relevant_parts[-1] if relevant_parts else 'doc'
        hash_suffix = hashlib.md5(source_file_path.encode()).hexdigest()[:8]
        doc_id = f"{base}_{hash_suffix}"
    return doc_id


def get_local_chunks_path(source_file_path: str, base_output: str) -> Path:
    path_without_ext = source_file_path.replace('.json', '')
    parts = Path(path_without_ext).parts
    skip_prefixes = {'raw', 'newapp', 'input', 'data'}
    relevant_parts = [p for p in parts if p.lower() not in skip_prefixes]
    if relevant_parts:
        directory_parts = relevant_parts[:-1]
        filename = relevant_parts[-1]
        if directory_parts:
            output_dir = Path(base_output) / "chunks" / Path(*directory_parts)
            return output_dir / f"{filename}_all_chunks.json"
        else:
            return Path(base_output) / "chunks" / f"{filename}_all_chunks.json"
    else:
        return Path(base_output) / "chunks" / "unknown_all_chunks.json"


def attach_same_role_chunk_ids(chunks: List[Dict]) -> None:
    role_to_ids: Dict = defaultdict(list)
    for chunk in chunks:
        role_to_ids[chunk.get('role', 'Others')].append(chunk['id'])
    for chunk in chunks:
        role = chunk.get('role', 'Others')
        chunk['same_role_chunk_ids'] = [
            cid for cid in role_to_ids[role] if cid != chunk['id']
        ]


class TestPipeline:
    """
    Test pipeline: fetches from ADLS, processes, saves outputs locally.
    Does NOT upload to ADLS or AI Search — use production_pipeline.py for that.
    """

    def __init__(self, doc_type: int, index_type: int, output_dir: str = "test_output"):
        logger.info("=" * 80)
        logger.info(
            f"TEST PIPELINE — "
            f"{DOC_TYPE_CONFIG[doc_type]['name']} / "
            f"{INDEX_TYPE_CONFIG[index_type]['name']}"
        )
        logger.info("=" * 80)

        self.doc_type = doc_type
        self.index_type = index_type
        self.adls_input_path = DOC_TYPE_CONFIG[doc_type]["adls_input_path"]
        self.role_weights = INDEX_TYPE_CONFIG[index_type]["role_weights"]

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.chunks_dir = self.output_dir / "chunks"
        self.chunks_dir.mkdir(exist_ok=True)

        self.stats_dir = self.output_dir / "stats"
        self.stats_dir.mkdir(exist_ok=True)

        logger.info(f"Output directory: {self.output_dir.absolute()}")

        self._validate_adls_config()
        self._init_adls()
        self._init_processors()

        self.role_classifier = None
        if ROLE_CLASSIFICATION_CONFIG["enabled"]:
            logger.info("Initializing role classifier...")
            self.role_classifier = create_classifier_from_config()
            if self.role_classifier:
                logger.info("Role classifier initialized successfully")
            else:
                logger.warning("Role classification disabled or failed to initialize")
        else:
            logger.info("Role classification disabled")

        logger.info("Test pipeline initialized successfully")

    def _validate_adls_config(self):
        errors = []
        if not ADLS_CONFIG["account_name"]:
            errors.append("ADLS_ACCOUNT_NAME not set")
        if not ADLS_CONFIG["account_key"]:
            errors.append("ADLS_ACCOUNT_KEY not set")
        if not ADLS_CONFIG["container_name"]:
            errors.append("ADLS_CONTAINER_NAME not set")
        if errors:
            raise ValueError("Configuration errors:\n" + "\n".join(f"  - {e}" for e in errors))

    def _init_adls(self):
        logger.info("Initializing ADLS connection...")
        self.adls_fetcher = ADLSFetcher(
            account_name=ADLS_CONFIG["account_name"],
            account_key=ADLS_CONFIG["account_key"],
            container_name=ADLS_CONFIG["container_name"]
        )

    def _init_processors(self):
        logger.info("Initializing text processors...")
        self.text_cleaner = LegalTextCleaner()
        self.semantic_chunker = SemanticChunker(
            model_name=EMBEDDING_CONFIG["model_name"],
            similarity_threshold=CHUNKING_CONFIG["similarity_threshold"],
            min_sentences_per_chunk=CHUNKING_CONFIG["min_sentences_per_chunk"],
            max_sentences_per_chunk=CHUNKING_CONFIG["max_sentences_per_chunk"],
            min_chunk_size=CHUNKING_CONFIG["min_chunk_size"]
        )
        self.embedding_model = self.semantic_chunker.model
        logger.info(f"Using embedding model: {EMBEDDING_CONFIG['model_name']}")

    def fetch_documents(self, max_docs: Optional[int] = None) -> List[Dict]:
        logger.info("\n" + "=" * 80)
        logger.info("STAGE 1: FETCHING DOCUMENTS FROM ADLS")
        logger.info("=" * 80)

        start_time = time.time()
        documents = self.adls_fetcher.fetch_all(
            path=self.adls_input_path,
            pattern=ADLS_CONFIG["file_pattern"],
            recursive=ADLS_CONFIG["recursive"],
            max_files=max_docs,
            show_progress=True
        )
        logger.info(f"Fetched {len(documents)} documents in {time.time() - start_time:.2f}s")
        return documents

    def process_single_document(
        self,
        doc: Dict
    ) -> Optional[Tuple[str, List[Dict], List[Dict], Path]]:
        source_file = doc.get("_source_file", "unknown.json")
        doc_id = generate_document_id_from_path(source_file)
        local_chunks_path = get_local_chunks_path(source_file, str(self.output_dir))
        relative_chunks_path = str(local_chunks_path.relative_to(self.output_dir))

        try:
            text = doc.get("full_text") or doc.get("judgment_text") or doc.get("text", "")
            if not text:
                logger.warning(f"Document {doc_id} has no text")
                return None

            cleaned_text = self.text_cleaner.clean(text)
            if not cleaned_text:
                logger.warning(f"Document {doc_id} is empty after cleaning")
                return None

            chunks, _ = self.semantic_chunker.split(
                cleaned_text,
                compute_doc_similarity=CHUNKING_CONFIG["compute_doc_similarity"]
            )

            if not chunks:
                logger.warning(f"Document {doc_id} produced no chunks")
                return None

            excluded_fields = {'text', 'full_text', 'judgment_text', 'embedding', '_source_file', 'raw_html_text'}
            metadata = {k: v for k, v in doc.items() if k not in excluded_fields}

            # Flatten nested 'metadata' dict; drop raw HTML markup and its truncated text duplicate
            if isinstance(metadata.get('metadata'), dict):
                nested = metadata.pop('metadata')
                nested_exclude = {'raw_html', 'description'}
                for k, v in nested.items():
                    if k not in nested_exclude:
                        metadata[k] = v

            all_chunks = []
            for idx, chunk in enumerate(chunks):
                chunk_all = {
                    "id": f"{doc_id}_{idx}",
                    "chunk_id": f"{doc_id}_{idx}",
                    "doc_id": doc_id,
                    "original_source_path": source_file,
                    "text": chunk["text"],
                    "doc_similarity": chunk.get("doc_similarity", 0.0),
                    "avg_similarity": chunk.get("avg_similarity", 0.0),
                    **metadata
                }
                all_chunks.append(chunk_all)

            if self.role_classifier:
                self.role_classifier.classify_chunks(
                    all_chunks,
                    text_field="text",
                    batch_size=ROLE_CLASSIFICATION_CONFIG["batch_size"],
                    add_to_chunks=True,
                    show_progress=False
                )
                for chunk in all_chunks:
                    if "role_prediction" in chunk:
                        chunk["role"] = chunk["role_prediction"]["role"]
                        chunk["confidence"] = chunk["role_prediction"]["confidence"]
                        del chunk["role_prediction"]

            attach_same_role_chunk_ids(all_chunks)

            if CHUNKING_CONFIG["top_k"]:
                selected_indices = weighted_topk_selection(
                    chunks=all_chunks,
                    top_k=CHUNKING_CONFIG["top_k"],
                    similarity_key="doc_similarity",
                    role_weights=self.role_weights
                )
            else:
                selected_indices = list(range(len(all_chunks)))

            top_k_chunks = []
            for idx in selected_indices:
                chunk_topk = all_chunks[idx].copy()
                chunk_topk["all_chunks_path"] = relative_chunks_path
                top_k_chunks.append(chunk_topk)

            return (doc_id, all_chunks, top_k_chunks, local_chunks_path)

        except Exception as e:
            logger.error(f"Error processing document {doc_id}: {e}", exc_info=True)
            return None

    def process_documents(
        self,
        documents: List[Dict]
    ) -> Tuple[Dict[str, Tuple[List[Dict], Path]], List[Dict], Dict]:
        logger.info("\n" + "=" * 80)
        logger.info("STAGE 2: PROCESSING DOCUMENTS")
        logger.info("=" * 80)

        start_time = time.time()
        doc_chunks_map = {}
        all_top_k_chunks = []
        failed_count = 0
        skipped_count = 0

        for doc in tqdm(documents, desc="Processing documents"):
            source_file = doc.get("_source_file", "unknown.json")
            local_chunks_path = get_local_chunks_path(source_file, str(self.output_dir))

            if local_chunks_path.exists():
                skipped_count += 1
                continue

            result = self.process_single_document(doc)

            if result is None:
                failed_count += 1
                if not PROCESSING_CONFIG["skip_errors"]:
                    raise ValueError("Document processing failed")
                continue

            doc_id, all_chunks, top_k_chunks, chunks_path = result
            doc_chunks_map[doc_id] = (all_chunks, chunks_path)
            all_top_k_chunks.extend(top_k_chunks)

        stats = {
            "total_documents": len(documents),
            "skipped_documents": skipped_count,
            "successful_documents": len(doc_chunks_map),
            "failed_documents": failed_count,
            "total_chunks": sum(len(chunks) for chunks, _ in doc_chunks_map.values()),
            "top_k_chunks": len(all_top_k_chunks),
            "processing_time": time.time() - start_time
        }

        logger.info(f"Processed {stats['successful_documents']}/{stats['total_documents']} documents")
        logger.info(f"Total chunks: {stats['total_chunks']}, Top-K: {stats['top_k_chunks']}")
        return doc_chunks_map, all_top_k_chunks, stats

    def generate_embeddings(self, chunks: List[Dict]) -> List[Dict]:
        logger.info("\n" + "=" * 80)
        logger.info("STAGE 3: GENERATING EMBEDDINGS")
        logger.info("=" * 80)

        if not chunks:
            logger.warning("No chunks to embed")
            return []

        start_time = time.time()
        texts = [chunk["text"] for chunk in chunks]

        logger.info(f"Encoding {len(texts)} texts...")
        embeddings = self.embedding_model.encode(
            texts,
            batch_size=EMBEDDING_CONFIG["batch_size"],
            show_progress_bar=True,
            convert_to_numpy=True
        )

        chunks_with_embeddings = []
        for chunk, embedding in zip(chunks, embeddings):
            chunk_copy = chunk.copy()
            chunk_copy["embedding"] = embedding.tolist()
            chunks_with_embeddings.append(chunk_copy)

        logger.info(f"Generated {len(embeddings)} embeddings in {time.time() - start_time:.2f}s")
        return chunks_with_embeddings

    def save_local_outputs(
        self,
        doc_chunks_map: Dict[str, Tuple[List[Dict], Path]],
        top_k_chunks: Optional[List[Dict]] = None,
        stats: Optional[Dict] = None
    ):
        logger.info("\n" + "=" * 80)
        logger.info("STAGE 4: SAVING OUTPUTS LOCALLY")
        logger.info("=" * 80)

        saved_count = 0
        for doc_id, (chunks, chunks_path) in tqdm(doc_chunks_map.items(), desc="Saving all_chunks files"):
            try:
                chunks_path.parent.mkdir(parents=True, exist_ok=True)
                safe_json_dump(chunks, chunks_path)
                saved_count += 1
            except Exception as e:
                logger.error(f"Failed to save chunks for {doc_id}: {e}")

        logger.info(f"Saved {saved_count}/{len(doc_chunks_map)} all_chunks files")

        if top_k_chunks:
            top_k_path = self.chunks_dir / "combined_top_k_chunks.json"
            try:
                safe_json_dump(top_k_chunks, top_k_path)
                logger.info(f"Saved {len(top_k_chunks)} top-k chunks to: {top_k_path}")
            except Exception as e:
                logger.error(f"Failed to save top-k chunks: {e}")

        if stats:
            stats_path = self.stats_dir / "pipeline_stats.json"
            try:
                safe_json_dump(stats, stats_path)
                logger.info(f"Saved statistics to: {stats_path}")
            except Exception as e:
                logger.error(f"Failed to save statistics: {e}")

        try:
            index = {
                "total_documents": len(doc_chunks_map),
                "documents": {
                    doc_id: {
                        "num_chunks": len(chunks),
                        "chunks_file": str(chunks_path.relative_to(self.output_dir))
                    }
                    for doc_id, (chunks, chunks_path) in doc_chunks_map.items()
                }
            }
            index_path = self.output_dir / "document_index.json"
            safe_json_dump(index, index_path)
            logger.info(f"Saved document index to: {index_path}")
        except Exception as e:
            logger.error(f"Failed to save document index: {e}")

    def run(self, max_documents: Optional[int] = None) -> Dict:
        pipeline_start = time.time()

        try:
            documents = self.fetch_documents(max_docs=max_documents)
            if not documents:
                logger.error("No documents fetched from ADLS")
                return {"error": "No documents found"}

            doc_chunks_map, top_k_chunks, proc_stats = self.process_documents(documents)
            if not doc_chunks_map:
                logger.error("No chunks generated")
                return {"error": "Processing failed"}

            top_k_with_embeddings = self.generate_embeddings(top_k_chunks) if top_k_chunks else []

            stats = {
                "status": "success",
                "mode": "test",
                "pipeline_time_seconds": round(time.time() - pipeline_start, 2),
                "documents_fetched": len(documents),
                "documents_processed": proc_stats["successful_documents"],
                "documents_failed": proc_stats["failed_documents"],
                "total_chunks_generated": proc_stats["total_chunks"],
                "top_k_chunks_selected": proc_stats["top_k_chunks"],
                "chunks_with_embeddings": len(top_k_with_embeddings),
                "timestamp": datetime.now().isoformat(),
                "output_directory": str(self.output_dir.absolute())
            }

            self.save_local_outputs(doc_chunks_map, top_k_with_embeddings, stats)

            logger.info("\n" + "=" * 80)
            logger.info("TEST PIPELINE COMPLETED SUCCESSFULLY")
            logger.info("=" * 80)

            return stats

        except Exception as e:
            logger.error(f"Test pipeline failed: {e}", exc_info=True)
            return {
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }


def setup_logging():
    log_format = LOGGING_CONFIG["format"]
    log_level = getattr(logging, LOGGING_CONFIG["level"].upper())

    handlers: list[logging.Handler] = [logging.StreamHandler()]

    Path("test_output").mkdir(exist_ok=True)
    handlers.append(logging.FileHandler("test_output/test_pipeline.log"))

    logging.basicConfig(level=log_level, format=log_format, handlers=handlers, force=True)


def main():
    parser = argparse.ArgumentParser(description="Test pipeline for legal document indexing")
    parser.add_argument("--doc_type",   type=int, choices=[0, 1], required=True,
                        help="Document type: 0 = High Court, 1 = Supreme Court")
    parser.add_argument("--index_type", type=int, choices=[0, 1], required=True,
                        help="Index type: 0 = AI Assistant, 1 = Precedent Finder")
    args = parser.parse_args()

    setup_logging()
    logger.info("Starting Test Pipeline")

    try:
        pipeline = TestPipeline(
            doc_type=args.doc_type,
            index_type=args.index_type,
            output_dir="test_output"
        )
        stats = pipeline.run(max_documents=PIPELINE_CONFIG["max_documents"] or 10)

        print("\n" + "=" * 80)
        print("TEST PIPELINE SUMMARY")
        print("=" * 80)
        print(json.dumps(stats, indent=2))
        print("=" * 80)

        exit(0 if stats.get("status") == "success" else 1)

    except KeyboardInterrupt:
        logger.warning("Test pipeline interrupted by user")
        exit(130)

    except Exception as e:
        logger.error(f"Test pipeline failed: {e}", exc_info=True)
        exit(1)


if __name__ == "__main__":
    main()

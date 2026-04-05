"""
Production Pipeline for Legal Document Processing
"""

import json
import logging
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from collections import defaultdict
import hashlib
import sys

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    LOGGING_CONFIG,
    ADLS_CONFIG,
    SEARCH_CONFIG,
    EMBEDDING_CONFIG,
    CHUNKING_CONFIG,
    PROCESSING_CONFIG,
    ROLE_CLASSIFICATION_CONFIG,
    PIPELINE_CONFIG,
    validate_config
)

from core.adls_fetcher import ADLSFetcher
from core.adls_uploader import ADLSUploader
from core.legal_text_cleaner import LegalTextCleaner
from core.semantic_chunker import SemanticChunker
from core.search_uploader import SearchIndexManager, SearchUploader
from tqdm import tqdm

from utils.weighted_selector import weighted_topk_selection
from core.role_classifier_future import create_classifier_from_config

logger = logging.getLogger(__name__)


# --------------------------------------------------
# UTILITIES
# --------------------------------------------------

def generate_document_id_from_path(source_file_path: str) -> str:

    path_without_ext = source_file_path.replace('.json', '')
    parts = Path(path_without_ext).parts

    skip_prefixes = {'raw', 'newapp', 'input', 'data'}
    relevant_parts = [p for p in parts if p.lower() not in skip_prefixes]

    doc_id = "_".join(relevant_parts).replace("-", "_").replace(" ", "_").lower()

    if len(doc_id) > 200:
        base = relevant_parts[-1] if relevant_parts else "doc"
        hash_suffix = hashlib.md5(source_file_path.encode()).hexdigest()[:8]
        doc_id = f"{base}_{hash_suffix}"

    return doc_id


def get_all_chunks_adls_path(source_file_path: str, base_output="processed") -> str:

    path_without_ext = source_file_path.replace('.json', '')
    parts = Path(path_without_ext).parts

    skip_prefixes = {'raw', 'newapp', 'input', 'data'}
    relevant_parts = [p for p in parts if p.lower() not in skip_prefixes]

    if relevant_parts:
        directory_parts = relevant_parts[:-1]
        filename = relevant_parts[-1]

        if directory_parts:
            output_dir = "/".join(directory_parts)
            return f"{base_output}/{output_dir}/{filename}_all_chunks.json"
        else:
            return f"{base_output}/{filename}_all_chunks.json"

    return f"{base_output}/unknown_all_chunks.json"


def attach_same_role_chunk_ids(chunks: List[Dict]):

    role_to_ids = defaultdict(list)

    for chunk in chunks:
        role_to_ids[chunk.get("role", "Others")].append(chunk["id"])

    for chunk in chunks:
        role = chunk.get("role", "Others")

        chunk["same_role_chunk_ids"] = [
            cid for cid in role_to_ids[role] if cid != chunk["id"]
        ]


# --------------------------------------------------
# PIPELINE
# --------------------------------------------------

class ProductionPipeline:

    def __init__(self, base_output_path="processed"):

        logger.info("=" * 80)
        logger.info("PRODUCTION PIPELINE")
        logger.info("=" * 80)

        validate_config()

        self.base_output_path = base_output_path

        self._init_adls()
        self._init_processors()
        self._init_search()

    # --------------------------------------------------

    def _init_adls(self):

        self.adls_fetcher = ADLSFetcher(
            account_name=ADLS_CONFIG["account_name"],
            account_key=ADLS_CONFIG["account_key"],
            container_name=ADLS_CONFIG["container_name"]
        )

        self.adls_uploader = ADLSUploader(
            account_name=ADLS_CONFIG["account_name"],
            account_key=ADLS_CONFIG["account_key"],
            container_name=ADLS_CONFIG["container_name"]
        )

    # --------------------------------------------------

    def _init_processors(self):

        self.text_cleaner = LegalTextCleaner()

        self.semantic_chunker = SemanticChunker(
            model_name=EMBEDDING_CONFIG["model_name"],
            similarity_threshold=CHUNKING_CONFIG["similarity_threshold"],
            min_sentences_per_chunk=CHUNKING_CONFIG["min_sentences_per_chunk"],
            max_sentences_per_chunk=CHUNKING_CONFIG["max_sentences_per_chunk"],
            min_chunk_size=CHUNKING_CONFIG["min_chunk_size"]
        )

        self.embedding_model = self.semantic_chunker.model

        self.role_classifier = None

        if ROLE_CLASSIFICATION_CONFIG["enabled"]:
            logger.info("Initializing role classifier")
            self.role_classifier = create_classifier_from_config()

    # --------------------------------------------------

    def _init_search(self):

        self.index_manager = SearchIndexManager(
            endpoint=SEARCH_CONFIG["endpoint"],
            key=SEARCH_CONFIG["key"]
        )

        self.search_uploader = SearchUploader(
            endpoint=SEARCH_CONFIG["endpoint"],
            key=SEARCH_CONFIG["key"],
            index_name=SEARCH_CONFIG["index_name"],
            batch_size=SEARCH_CONFIG["upload_batch_size"],
            max_retries=SEARCH_CONFIG["max_retries"],
            retry_delay=SEARCH_CONFIG["retry_delay"]
        )

    # --------------------------------------------------

    def fetch_documents(self):

        documents = self.adls_fetcher.fetch_all(
            path=ADLS_CONFIG["input_path"],
            pattern=ADLS_CONFIG["file_pattern"],
            recursive=ADLS_CONFIG["recursive"],
            max_files=PIPELINE_CONFIG["max_documents"],
            show_progress=True
        )

        return documents

    # --------------------------------------------------

    def process_single_document(self, doc: Dict):

        source_file = doc.get("_source_file", "unknown.json")

        doc_id = generate_document_id_from_path(source_file)

        all_chunks_path = get_all_chunks_adls_path(source_file, self.base_output_path)

        try:

            text = doc.get("text", "")

            if not text:
                return None

            cleaned_text = self.text_cleaner.clean(text)

            chunks, _ = self.semantic_chunker.split(
                cleaned_text,
                compute_doc_similarity=CHUNKING_CONFIG["compute_doc_similarity"]
            )

            excluded_fields = {"text", "embedding", "_source_file"}
            metadata = {k: v for k, v in doc.items() if k not in excluded_fields}

            all_chunks = []

            for idx, chunk in enumerate(chunks):

                chunk_all = {
                    "id": f"{doc_id}_{idx}",
                    "doc_id": doc_id,
                    "original_source_path": source_file,
                    "text": chunk["text"],
                    **metadata
                }

                all_chunks.append(chunk_all)

            # ROLE CLASSIFICATION
            if self.role_classifier:

                self.role_classifier.classify_chunks(
                    all_chunks,
                    text_field="text",
                    batch_size=ROLE_CLASSIFICATION_CONFIG["batch_size"],
                    add_to_chunks=True,
                    show_progress=False
                )

            attach_same_role_chunk_ids(all_chunks)

            # TOP-K SELECTION
            if CHUNKING_CONFIG["top_k"]:

                selected_indices = weighted_topk_selection(
                    chunks=all_chunks,
                    top_k=CHUNKING_CONFIG["top_k"],
                    similarity_key="doc_similarity"
                )

            else:
                selected_indices = list(range(len(all_chunks)))

            top_k_chunks = []

            for idx in selected_indices:

                chunk_topk = all_chunks[idx].copy()
                chunk_topk["all_chunks_path"] = all_chunks_path

                top_k_chunks.append(chunk_topk)

            return (doc_id, all_chunks, top_k_chunks, all_chunks_path)

        except Exception as e:

            logger.error(f"Error processing document {doc_id}: {e}")

            return None

    # --------------------------------------------------

    def process_documents(self, documents):

        doc_chunks_map = {}
        all_top_k_chunks = []

        for doc in tqdm(documents):

            result = self.process_single_document(doc)

            if result is None:
                continue

            doc_id, all_chunks, top_k_chunks, chunks_path = result

            doc_chunks_map[doc_id] = {
                "chunks": all_chunks,
                "path": chunks_path
            }

            all_top_k_chunks.extend(top_k_chunks)

        stats = {
            "total_documents": len(documents),
            "successful_documents": len(doc_chunks_map),
            "total_chunks": sum(len(v["chunks"]) for v in doc_chunks_map.values()),
            "top_k_chunks": len(all_top_k_chunks)
        }

        return doc_chunks_map, all_top_k_chunks, stats

    # --------------------------------------------------

    def generate_embeddings(self, chunks):

        texts = [chunk["text"] for chunk in chunks]

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

        return chunks_with_embeddings

    # --------------------------------------------------

    def upload_chunks_to_adls(self, doc_chunks_map):

        upload_results = {}

        for doc_id, doc_data in doc_chunks_map.items():

            success = self.adls_uploader.upload_json_file(
                data=doc_data["chunks"],
                adls_path=doc_data["path"],
                overwrite=True
            )

            upload_results[doc_id] = success

        return upload_results

    # --------------------------------------------------

    def upload_topk_to_search(self, top_k_chunks):

        stats = self.search_uploader.upload_chunks(
            chunks=top_k_chunks,
            show_progress=True
        )

        return stats

    # --------------------------------------------------

    def run(self):

        start = time.time()

        documents = self.fetch_documents()

        doc_chunks_map, top_k_chunks, proc_stats = self.process_documents(documents)

        top_k_with_embeddings = self.generate_embeddings(top_k_chunks)

        upload_results = self.upload_chunks_to_adls(doc_chunks_map)

        if PIPELINE_CONFIG["create_index"]:
            self.index_manager.create_legal_documents_index(
                index_name=SEARCH_CONFIG["index_name"],
                vector_dimensions=EMBEDDING_CONFIG["dimensions"]
            )

        if PIPELINE_CONFIG["upload_to_search"]:
            search_stats = self.upload_topk_to_search(top_k_with_embeddings)
        else:
            search_stats = {}

        pipeline_time = time.time() - start

        stats = {
            "status": "success",
            "pipeline_time_seconds": round(pipeline_time, 2),
            "documents_processed": proc_stats["successful_documents"],
            "total_chunks": proc_stats["total_chunks"],
            "top_k_chunks": proc_stats["top_k_chunks"],
            "search_upload": search_stats
        }

        logger.info("PIPELINE COMPLETE")

        return stats


# --------------------------------------------------

def setup_logging():

    log_format = LOGGING_CONFIG["format"]
    log_level = getattr(logging, LOGGING_CONFIG["level"].upper())

    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[logging.StreamHandler()],
        force=True
    )


def main():

    setup_logging()

    pipeline = ProductionPipeline(base_output_path="processed")

    stats = pipeline.run()

    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
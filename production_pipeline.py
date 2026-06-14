"""
Production Pipeline for Legal Document Processing

Usage:
    python production_pipeline.py --doc_type 1 --index_type 0
        doc_type:   0 = High Court,   1 = Supreme Court
        index_type: 0 = AI Assistant, 1 = Precedent Finder

Architecture (producer/consumer, 7 stages):
    Stage 1  ThreadPool  IO       Fetch + clean text
    Stage 2  ThreadPool  CPU      Split sentences
    Stage 3  GPU thread  GPU      Batch-encode sentences across docs
    Stage 4  CPU thread  CPU      Assemble SemanticChunk objects
    Stage 5  GPU thread  GPU      Batch role-classify all chunk texts
    Stage 6  GPU thread  GPU      Batch final embedding for Search upload
    Stage 7  ThreadPool  IO       ADLS upload + Search upload
"""
import argparse
import json
import logging
import os
import queue
import threading
import time
import hashlib
import numpy as np

# Suppress Azure SDK per-request HTTP traces and urllib3 pool-full noise at import time
logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(logging.WARNING)
logging.getLogger("azure.storage").setLevel(logging.WARNING)
logging.getLogger("azure.core").setLevel(logging.WARNING)
logging.getLogger("urllib3.connectionpool").setLevel(logging.ERROR)
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

from config import (
    LOGGING_CONFIG,
    ADLS_CONFIG,
    SEARCH_CONFIG,
    EMBEDDING_CONFIG,
    CHUNKING_CONFIG,
    PROCESSING_CONFIG,
    ROLE_CLASSIFICATION_CONFIG,
    PIPELINE_CONFIG,
    DOC_TYPE_CONFIG,
    INDEX_TYPE_CONFIG,
    validate_config
)

from core.adls_fetcher import ADLSFetcher
from core.adls_uploader import ADLSUploader
from core.legal_text_cleaner import LegalTextCleaner
from core.semantic_chunker import SemanticChunker
from core.search_uploader import SearchIndexManager, SearchUploader
from core.role_classifier import create_classifier_from_config
from utils.weighted_selector import weighted_topk_selection
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Sentinel used to signal workers that the upstream stage is done.
_STOP = object()


# --------------------------------------------------
# UTILITIES
# --------------------------------------------------

def generate_document_id_from_path(source_file_path: str) -> str:
    path_without_ext = source_file_path.replace('.json', '')
    parts = Path(path_without_ext).parts
    skip_prefixes = {'raw', 'newapp', 'input', 'data', 'app'}
    relevant_parts = [p for p in parts if p.lower() not in skip_prefixes]
    doc_id = "_".join(relevant_parts).replace("-", "_").replace(" ", "_").lower()
    if len(doc_id) > 200:
        base = relevant_parts[-1] if relevant_parts else "doc"
        hash_suffix = hashlib.md5(source_file_path.encode()).hexdigest()[:8]
        doc_id = f"{base}_{hash_suffix}"
    return doc_id


def get_all_chunks_adls_path(source_file_path: str, base_output: str = "processed") -> str:
    path_without_ext = source_file_path.replace('.json', '')
    parts = Path(path_without_ext).parts
    skip_prefixes = {'raw', 'newapp', 'input', 'data', 'app'}
    relevant_parts = [p for p in parts if p.lower() not in skip_prefixes]
    if relevant_parts:
        directory_parts = relevant_parts[:-1]
        filename = relevant_parts[-1]
        if directory_parts:
            output_dir = "/".join(directory_parts)
            return f"{base_output}/{output_dir}/{filename}_all_chunks.json"
        return f"{base_output}/{filename}_all_chunks.json"
    return f"{base_output}/unknown_all_chunks.json"


def get_done_marker_path(source_file_path: str, base_output: str = "processed", index_type: int = 0) -> str:
    path_without_ext = source_file_path.replace('.json', '')
    parts = Path(path_without_ext).parts
    skip_prefixes = {'raw', 'newapp', 'input', 'data', 'app'}
    relevant_parts = [p for p in parts if p.lower() not in skip_prefixes]
    if relevant_parts:
        directory_parts = relevant_parts[:-1]
        filename = relevant_parts[-1]
        if directory_parts:
            output_dir = "/".join(directory_parts)
            return f"{base_output}/{output_dir}/{filename}_done_{index_type}.json"
        return f"{base_output}/{filename}_done_{index_type}.json"
    return f"{base_output}/unknown_done_{index_type}.json"


def attach_same_role_chunk_ids(chunks: List[Dict]) -> None:
    role_to_ids: Dict = defaultdict(list)
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

    def __init__(self, doc_type: int, index_type: int, base_output_path: str = "processed",
                 start_year: Optional[int] = None, end_year: Optional[int] = None):
        logger.info("=" * 80)
        logger.info(
            f"PRODUCTION PIPELINE — "
            f"{DOC_TYPE_CONFIG[doc_type]['name']} / "
            f"{INDEX_TYPE_CONFIG[index_type]['name']}"
        )
        logger.info("=" * 80)

        self.doc_type = doc_type
        self.index_type = index_type
        self.base_output_path = base_output_path
        self.start_year = start_year
        self.end_year = end_year

        if start_year or end_year:
            logger.info(f"Year filter: {start_year or '–'} → {end_year or '–'}")

        self.adls_input_path = DOC_TYPE_CONFIG[doc_type]["adls_input_path"]
        self.index_name = DOC_TYPE_CONFIG[doc_type]["index_names"][index_type]
        self.role_weights = INDEX_TYPE_CONFIG[index_type]["role_weights"]

        validate_config()

        self._init_adls()
        self._init_processors()
        self._init_search()

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

    def _init_search(self):
        self.index_manager = SearchIndexManager(
            endpoint=SEARCH_CONFIG["endpoint"],
            key=SEARCH_CONFIG["key"]
        )
        self.search_uploader = SearchUploader(
            endpoint=SEARCH_CONFIG["endpoint"],
            key=SEARCH_CONFIG["key"],
            index_name=self.index_name,
            doc_type=self.doc_type,
            batch_size=SEARCH_CONFIG["upload_batch_size"],
            max_retries=SEARCH_CONFIG["max_retries"],
            retry_delay=SEARCH_CONFIG["retry_delay"]
        )

    def _stream_into_pipeline(
        self, pending_q: "queue.Queue"
    ) -> Tuple[List[Tuple], List[int], threading.Thread]:
        """
        Streams file listing in batches of BATCH_SIZE from ADLS.

        For each batch:
          1. Check all files for done/resume markers in parallel (CHECK_WORKERS).
          2. Read JSON for unprocessed files in parallel (io_workers).
          3. Push pending docs into pending_q.

        A lister thread builds batches while the checker thread processes the
        previous batch — they run concurrently via a 2-slot batch_q so listing
        stays at most one batch ahead.

        Returns (resume_candidates, skipped_ref, thread).
        """
        BATCH_SIZE    = int(os.getenv("STREAM_BATCH_SIZE", "30000"))
        CHECK_WORKERS = int(os.getenv("CHECK_WORKERS", "400"))   # pure HTTP existence checks
        io_workers    = PIPELINE_CONFIG.get("io_workers", 64)
        max_docs     = PIPELINE_CONFIG.get("max_documents", None)

        resume_candidates: List[Tuple] = []
        skipped_ref = [0]
        batch_q: "queue.Queue" = queue.Queue(maxsize=2)   # lister stays ≤1 batch ahead

        def _check_one(fp: str):
            all_chunks_path  = get_all_chunks_adls_path(fp, self.base_output_path)
            done_marker_path = get_done_marker_path(fp, self.base_output_path, self.index_type)
            if done_marker_path in self._processed_cache:
                return "skip", None
            if all_chunks_path in self._processed_cache:
                return "resume", (fp, all_chunks_path, done_marker_path)
            return "needs_read", fp

        def _read_one(fp: str):
            try:
                doc = self.adls_fetcher.read_json_file(fp)
                if doc is None:
                    return None
                doc["_source_file"] = fp
                return doc
            except Exception as e:
                logger.warning(f"Read error {fp}: {e}")
                return None

        def _process_batch(fps: List[str]):
            needs_read: List[str] = []
            with ThreadPoolExecutor(max_workers=CHECK_WORKERS) as pool:
                futures = [pool.submit(_check_one, fp) for fp in fps]
                for f in as_completed(futures):
                    try:
                        result_type, payload = f.result()
                    except Exception as e:
                        logger.warning(f"Check error: {e}")
                        skipped_ref[0] += 1
                        continue
                    if result_type == "skip":
                        skipped_ref[0] += 1
                    elif result_type == "resume":
                        resume_candidates.append(payload)
                    else:
                        needs_read.append(payload)

            with ThreadPoolExecutor(max_workers=io_workers) as pool:
                futures = [pool.submit(_read_one, fp) for fp in needs_read]
                for f in as_completed(futures):
                    try:
                        doc = f.result()
                    except Exception as e:
                        logger.warning(f"Read error: {e}")
                        skipped_ref[0] += 1
                        continue
                    if doc is None:
                        skipped_ref[0] += 1
                    else:
                        pending_q.put(doc)

            logger.info(
                f"Batch done: {len(fps)} checked, {len(needs_read)} pending, "
                f"resume={len(resume_candidates)}, skipped={skipped_ref[0]}"
            )

        def _lister():
            batch: List[str] = []
            seen = 0

            # Build list of paths to iterate — one per year if filtered, else root.
            if self.start_year or self.end_year:
                sy = self.start_year or 1900
                ey = self.end_year   or 2100
                year_paths = [
                    f"{self.adls_input_path.rstrip('/')}/year={yr}"
                    for yr in range(sy, ey + 1)
                ]
            else:
                year_paths = [self.adls_input_path]

            for ypath in year_paths:
                file_iter = self.adls_fetcher.list_files_iter(
                    path=ypath,
                    pattern=ADLS_CONFIG["file_pattern"],
                    recursive=ADLS_CONFIG["recursive"]
                )
                for fp in file_iter:
                    if max_docs and seen >= max_docs:
                        batch_q.put(batch or [])
                        batch_q.put(_STOP)
                        return
                    batch.append(fp)
                    seen += 1
                    if len(batch) >= BATCH_SIZE:
                        batch_q.put(batch)
                        batch = []
            if batch:
                batch_q.put(batch)
            batch_q.put(_STOP)

        def _checker():
            while True:
                batch = batch_q.get()
                if batch is _STOP:
                    pending_q.put(_STOP)
                    return
                _process_batch(batch)

        def _worker():
            t_list  = threading.Thread(target=_lister,  name="lister",  daemon=True)
            t_check = threading.Thread(target=_checker, name="checker", daemon=True)
            t_list.start()
            t_check.start()
            t_check.join()

        t = threading.Thread(target=_worker, name="stream-fetch", daemon=True)
        t.start()
        return resume_candidates, skipped_ref, t

    def generate_embeddings(self, chunks: List[Dict]) -> List[Dict]:
        texts = [chunk["text"] for chunk in chunks]
        embeddings = self.semantic_chunker.encode_batch(texts)
        chunks_with_embeddings = []
        for chunk, emb in zip(chunks, embeddings):
            c = chunk.copy()
            c["embedding"] = emb.tolist()
            chunks_with_embeddings.append(c)
        return chunks_with_embeddings

    def upload_chunks_to_adls(self, doc_chunks_map: Dict) -> Dict:
        results = {}
        for doc_id, doc_data in doc_chunks_map.items():
            results[doc_id] = self.adls_uploader.upload_json_file(
                data=doc_data["chunks"],
                adls_path=doc_data["path"],
                overwrite=True
            )
        return results

    def upload_topk_to_search(self, top_k_chunks: List[Dict]) -> Dict:
        return self.search_uploader.upload_chunks(top_k_chunks, show_progress=True)

    def _resume_pending_search(self, resume_candidates: List[Tuple]) -> int:
        if not resume_candidates:
            return 0
        total = len(resume_candidates)
        logger.info(f"Resuming search upload for {total} pending documents")
        # Use fewer workers to avoid ADLS throttling (especially when two pipelines run simultaneously)
        io_workers = min(PIPELINE_CONFIG.get("io_workers", 64), 32)
        completed_ref = [0]
        skipped_ref = [0]
        fetched_ref = [0]

        EMBED_BATCH = 2048

        pending_embed: List[Dict] = []
        pending_meta:  List[Tuple] = []

        def flush_embed_batch():
            if not pending_embed:
                return
            n_docs = len(pending_meta)
            n_chunks = len(pending_embed)
            texts = [c["text"] for c in pending_embed]
            logger.info(
                f"[Resume] GPU embed: {n_chunks} chunks from {n_docs} docs "
                f"(uploaded={completed_ref[0]}, skipped={skipped_ref[0]}, fetched={fetched_ref[0]}/{total})"
            )
            try:
                embs = self.semantic_chunker.encode_batch(texts)
                for chunk, emb in zip(pending_embed, embs):
                    chunk["embedding"] = emb.tolist()
            except Exception as e:
                logger.error(f"[Resume] Embedding batch failed, skipping {n_docs} docs: {e}")
                n = len(pending_meta)
                pending_embed.clear()
                pending_meta.clear()
                skipped_ref[0] += n
                return

            # Upload all chunks in one batched call, then write done markers
            try:
                stats = self.search_uploader.upload_chunks(list(pending_embed), show_progress=False)
                batch_failed = stats.get("failed", 0) > 0
            except Exception as e:
                logger.warning(f"[Resume] Upload failed for batch: {e}")
                skipped_ref[0] += len(pending_meta)
                pending_embed.clear()
                pending_meta.clear()
                return

            if batch_failed:
                skipped_ref[0] += len(pending_meta)
            else:
                marker_paths = [p for p, _ in pending_meta]
                with ThreadPoolExecutor(max_workers=io_workers) as marker_pool:
                    marker_futures = {marker_pool.submit(self.adls_uploader.write_marker, p): p for p in marker_paths}
                    for fut in as_completed(marker_futures):
                        try:
                            fut.result()
                            completed_ref[0] += 1
                        except Exception as e:
                            logger.warning(f"[Resume] Marker write failed: {e}")
                            skipped_ref[0] += 1

            logger.info(
                f"[Resume] Batch done — uploaded={completed_ref[0]}, "
                f"skipped={skipped_ref[0]}, remaining={total - fetched_ref[0]}"
            )
            pending_embed.clear()
            pending_meta.clear()

        def process_one(candidate):
            source_file, all_chunks_path, done_marker_path = candidate
            try:
                all_chunks_data = self.adls_fetcher.read_json_file(all_chunks_path)
            except Exception as e:
                logger.warning(f"[Resume] Skipping {source_file}: cannot read chunks ({e})")
                return None
            all_chunks = all_chunks_data if isinstance(all_chunks_data, list) else []
            if not all_chunks:
                logger.warning(f"[Resume] Skipping {source_file}: chunks file empty")
                return None
            if CHUNKING_CONFIG["top_k"]:
                selected = weighted_topk_selection(
                    all_chunks, CHUNKING_CONFIG["top_k"], "doc_similarity", self.role_weights
                )
            else:
                selected = list(range(len(all_chunks)))
            top_k = [all_chunks[i] for i in selected]
            return done_marker_path, top_k

        LOG_EVERY = 500
        with ThreadPoolExecutor(max_workers=io_workers) as pool:
            futures = {pool.submit(process_one, c): c for c in resume_candidates}
            with tqdm(total=total, desc="Resuming search upload", position=0, leave=True) as pbar:
                for future in as_completed(futures):
                    pbar.update(1)
                    fetched_ref[0] += 1
                    try:
                        result = future.result()
                    except Exception as e:
                        logger.warning(f"[Resume] Fetch error, skipping: {e}")
                        skipped_ref[0] += 1
                        continue
                    if result is None:
                        skipped_ref[0] += 1
                        continue
                    done_marker_path, top_k = result
                    pending_embed.extend(top_k)
                    pending_meta.append((done_marker_path, len(top_k)))
                    if fetched_ref[0] % LOG_EVERY == 0:
                        logger.info(
                            f"[Resume] Fetched {fetched_ref[0]}/{total}, "
                            f"pending_chunks={len(pending_embed)}, "
                            f"uploaded={completed_ref[0]}, skipped={skipped_ref[0]}"
                        )
                    if len(pending_embed) >= EMBED_BATCH:
                        flush_embed_batch()

        flush_embed_batch()

        logger.info(
            f"Resume complete — uploaded: {completed_ref[0]}, "
            f"skipped: {skipped_ref[0]}, total: {total}"
        )
        return completed_ref[0]

    def _write_done_markers(self, doc_chunks_map: Dict, adls_results: Dict, search_had_failures: bool) -> int:
        if search_had_failures:
            logger.warning("Search upload had failures — skipping _done markers (will resume on next run)")
            return 0
        written = 0
        for doc_id, doc_data in doc_chunks_map.items():
            if adls_results.get(doc_id):
                done_path = doc_data["path"].replace("_all_chunks.json", f"_done_{self.index_type}.json")
                if self.adls_uploader.write_marker(done_path):
                    written += 1
        logger.info(f"Wrote {written} _done markers")
        return written

    # ------------------------------------------------------------------
    # PRODUCER / CONSUMER PIPELINE
    # ------------------------------------------------------------------

    def _run_queue_pipeline(self, pending_docs, total_docs: int = None) -> Dict:
        """
        7-stage producer/consumer pipeline.

        pending_docs may be a List[Dict] or a queue.Queue (fed by the parallel status
        checker).  When it is a Queue, items arrive as they are confirmed pending and
        the queue is terminated with _STOP.

        Returns upload_stats dict with processed/total_chunks/total_top_k/search counts.
        GPU stages accumulate items from multiple documents before calling the GPU,
        so the GPU always sees large batches rather than one document at a time.
        """
        io_workers       = PIPELINE_CONFIG.get("io_workers", 64)
        assemble_workers = int(os.getenv("ASSEMBLE_WORKERS", "16"))  # Stage 4 thread pool
        # GPU accumulation threshold: accumulate this many sentences/chunks before
        # dispatching a GPU call. Sized to fill the A100 VRAM with useful work.
        SENT_BATCH  = int(os.getenv("SENT_BATCH",  "16384"))  # sentences for Stage 3
        CHUNK_BATCH = int(os.getenv("CHUNK_BATCH",  "8192"))  # chunks for Stage 5 + 6

        # ----- queues -----
        # Stage 2 → Stage 3: (doc_id, doc, sentences, cleaned_text, all_chunks_path)
        q_sentences: queue.Queue = queue.Queue(maxsize=2048)
        # Stage 3 → Stage 4: (doc_id, doc, sentences, sent_embs, cleaned_text, all_chunks_path)
        q_embedded: queue.Queue = queue.Queue(maxsize=1024)
        # Stage 4 → Stage 5: (doc_id, doc, all_chunks_path, chunks[SemanticChunk], chunk_texts[str])
        q_chunks: queue.Queue = queue.Queue(maxsize=1024)
        # Stage 5 → Stage 6: (doc_id, doc, all_chunks_path, chunks_with_roles[SemanticChunk or dict])
        q_roles: queue.Queue = queue.Queue(maxsize=1024)
        # Stage 6 → collector: (doc_id, all_chunks_path, all_chunks[dict], top_k_chunks[dict])
        q_final: queue.Queue = queue.Queue(maxsize=1024)

        # total_docs may be None when streaming from a queue (unknown upfront)
        if total_docs is None:
            total_docs = len(pending_docs) if not isinstance(pending_docs, queue.Queue) else None
        errors = []

        # ---- Stage 1+2: IO fetch is already done; just clean + split sentences ----
        def stage2_split(doc: Dict):
            """Called in thread pool. Cleans text, splits sentences, puts to q_sentences."""
            source_file = doc.get("_source_file", "unknown.json")
            doc_id = generate_document_id_from_path(source_file)
            all_chunks_path = get_all_chunks_adls_path(source_file, self.base_output_path)
            try:
                text = doc.get("full_text") or doc.get("judgment_text") or doc.get("text", "")
                if not text:
                    return
                cleaned_text = self.text_cleaner.clean(text)
                if not cleaned_text:
                    return
                sentences = self.semantic_chunker._split_sentences(cleaned_text)
                if not sentences:
                    return
                q_sentences.put((doc_id, doc, sentences, cleaned_text, all_chunks_path))
            except Exception as e:
                logger.error(f"Stage 2 error for {doc_id}: {e}", exc_info=True)
                errors.append(e)

        def stage2_producer():
            """Submits docs to thread pool. Accepts a list or a streaming Queue."""
            with ThreadPoolExecutor(max_workers=io_workers) as pool:
                pending_futs = []
                if isinstance(pending_docs, queue.Queue):
                    # Streaming mode: drain the queue until _STOP arrives
                    while True:
                        doc = pending_docs.get()
                        if doc is _STOP:
                            break
                        pending_futs.append(pool.submit(stage2_split, doc))
                else:
                    pending_futs = [pool.submit(stage2_split, doc) for doc in pending_docs]
                for f in as_completed(pending_futs):
                    exc = f.exception()
                    if exc:
                        errors.append(exc)
            q_sentences.put(_STOP)

        # ---- Stage 3: GPU batch sentence encoding ----
        def stage3_encode():
            """
            Drains q_sentences, accumulates up to SENT_BATCH sentences, then calls
            encode_batch() once. Splits the result back per document.
            """
            pending_items = []   # list of (doc_id, doc, sentences, cleaned_text, all_chunks_path)
            pending_counts = []  # number of sentences per pending doc

            def flush():
                if not pending_items:
                    return
                all_sents = []
                for _, _, sents, _, _ in pending_items:
                    all_sents.extend(sents)
                embs = self.semantic_chunker.encode_batch(all_sents)
                offset = 0
                for (did, d, sents, ct, acp), cnt in zip(pending_items, pending_counts):
                    doc_embs = embs[offset: offset + cnt]
                    offset += cnt
                    q_embedded.put((did, d, sents, doc_embs, ct, acp))
                pending_items.clear()
                pending_counts.clear()

            while True:
                item = q_sentences.get()
                if item is _STOP:
                    flush()
                    q_embedded.put(_STOP)
                    return
                doc_id, doc, sentences, cleaned_text, all_chunks_path = item
                pending_items.append((doc_id, doc, sentences, cleaned_text, all_chunks_path))
                pending_counts.append(len(sentences))
                if sum(pending_counts) >= SENT_BATCH:
                    flush()

        # ---- Stage 4: CPU chunk assembly (thread pool) ----
        def stage4_assemble():
            """
            CPU-only, pure numpy — fully thread-safe.
            Submitter feeds the pool as items arrive from q_embedded.
            Drainer forwards results to q_chunks as each future completes,
            without waiting for all of Stage 3 to finish first.
            """
            def _assemble_one(item):
                doc_id, doc, sentences, sent_embs, cleaned_text, all_chunks_path = item
                chunks, chunk_texts = self.semantic_chunker.assemble_chunks_cpu(
                    cleaned_text, sentences, sent_embs
                )
                if not chunks:
                    return None
                return (doc_id, doc, all_chunks_path, chunks, chunk_texts)

            # Bounded queue of futures: caps memory if drainer falls behind
            fut_q: queue.Queue = queue.Queue(maxsize=assemble_workers * 8)

            def _submitter():
                with ThreadPoolExecutor(max_workers=assemble_workers) as pool:
                    while True:
                        item = q_embedded.get()
                        if item is _STOP:
                            break
                        fut_q.put(pool.submit(_assemble_one, item))
                fut_q.put(_STOP)

            def _drainer():
                while True:
                    f = fut_q.get()
                    if f is _STOP:
                        q_chunks.put(_STOP)
                        return
                    try:
                        result = f.result()
                    except Exception as e:
                        logger.error(f"Stage 4 error: {e}", exc_info=True)
                        errors.append(e)
                        continue
                    if result is not None:
                        q_chunks.put(result)

            t_sub   = threading.Thread(target=_submitter, daemon=True)
            t_drain = threading.Thread(target=_drainer,   daemon=True)
            t_sub.start()
            t_drain.start()
            t_sub.join()
            t_drain.join()

        # ---- Stage 5: GPU batch role classification ----
        def stage5_classify():
            """
            Accumulates chunk texts across documents, role-classifies in one big
            batch, then distributes results back.
            """
            pending: List[Tuple] = []   # (doc_id, doc, acp, chunks, chunk_texts)
            pending_counts: List[int] = []

            def flush():
                if not pending:
                    return
                if self.role_classifier:
                    all_texts = []
                    for _, _, _, _, ctexts in pending:
                        all_texts.extend(ctexts)
                    preds = self.role_classifier.predict(
                        all_texts,
                        batch_size=ROLE_CLASSIFICATION_CONFIG["batch_size"],
                        return_probabilities=True
                    )
                    offset = 0
                    for (did, d, acp, chunks, ctexts), cnt in zip(pending, pending_counts):
                        doc_preds = preds[offset: offset + cnt]
                        offset += cnt
                        enriched = []
                        for chunk, pred in zip(chunks, doc_preds):
                            cd = chunk.to_dict() if hasattr(chunk, 'to_dict') else dict(chunk)
                            cd["role"] = pred["role"]
                            cd["confidence"] = float(pred["confidence"])
                            enriched.append(cd)
                        q_roles.put((did, d, acp, enriched))
                else:
                    for (did, d, acp, chunks, _ctexts) in pending:
                        plain = [
                            (chunk.to_dict() if hasattr(chunk, 'to_dict') else dict(chunk))
                            for chunk in chunks
                        ]
                        q_roles.put((did, d, acp, plain))
                pending.clear()
                pending_counts.clear()

            while True:
                item = q_chunks.get()
                if item is _STOP:
                    flush()
                    q_roles.put(_STOP)
                    return
                doc_id, doc, acp, chunks, chunk_texts = item
                pending.append((doc_id, doc, acp, chunks, chunk_texts))
                pending_counts.append(len(chunk_texts))
                if sum(pending_counts) >= CHUNK_BATCH:
                    flush()

        # ---- Stage 6: GPU batch final embedding (for Search) ----
        def stage6_embed():
            """
            Accumulates top-k chunk texts across docs, encodes in one batch, then
            splits embeddings back per document and pushes to q_final.
            """
            pending: List[Tuple] = []   # (doc_id, doc, acp, chunk_dicts)
            pending_topk_counts: List[int] = []

            def flush():
                if not pending:
                    return

                per_doc_all: List[List[Dict]] = []
                per_doc_topk: List[List[Dict]] = []

                for doc_id, doc, acp, chunk_dicts in pending:
                    source_file = doc.get("_source_file", "unknown.json")
                    excluded_fields = {"text", "full_text", "judgment_text", "embedding", "_source_file", "raw_html_text"}
                    metadata = {k: v for k, v in doc.items() if k not in excluded_fields}

                    if isinstance(metadata.get('metadata'), dict):
                        nested = metadata.pop('metadata')
                        nested_exclude = {'raw_html', 'description'}
                        for k, v in nested.items():
                            if k not in nested_exclude:
                                metadata[k] = v

                    resolved_date = (
                        metadata.get("date")
                        or metadata.get("decision_date", "")
                        or metadata.get("year", "")
                    )
                    config_jurisdiction = DOC_TYPE_CONFIG[self.doc_type].get("jurisdiction")
                    if config_jurisdiction:
                        resolved_jurisdiction = config_jurisdiction
                    else:
                        resolved_jurisdiction = (
                            metadata.get("jurisdiction")
                            or metadata.get("court", "")
                            or metadata.get("bench", "")
                        )

                    all_chunks_out = []
                    for idx, cd in enumerate(chunk_dicts):
                        chunk_all = {
                            "doc_id":               doc_id,
                            "date":                 resolved_date,
                            "jurisdiction":         resolved_jurisdiction,
                            **metadata,
                            "id":                   f"{doc_id}_{idx}",
                            "chunk_id":             f"{doc_id}_{idx}",
                            "original_source_path": source_file,
                            "text":                 cd["text"],
                            "start_char":           cd.get("start_char", 0),
                            "end_char":             cd.get("end_char", 0),
                            "num_sentences":        cd.get("num_sentences", 0),
                            "doc_similarity":       float(cd.get("doc_similarity", 0.0)),
                            "avg_similarity":       float(cd.get("avg_similarity", 0.0)),
                            "role":                 cd.get("role", "Others"),
                            "confidence":           float(cd.get("confidence", 0.0)),
                        }
                        all_chunks_out.append(chunk_all)

                    attach_same_role_chunk_ids(all_chunks_out)

                    if CHUNKING_CONFIG["top_k"]:
                        selected_indices = weighted_topk_selection(
                            chunks=all_chunks_out,
                            top_k=CHUNKING_CONFIG["top_k"],
                            similarity_key="doc_similarity",
                            role_weights=self.role_weights
                        )
                    else:
                        selected_indices = list(range(len(all_chunks_out)))

                    topk_out = []
                    for i in selected_indices:
                        c = all_chunks_out[i].copy()
                        c["all_chunks_path"] = acp
                        topk_out.append(c)

                    per_doc_all.append(all_chunks_out)
                    per_doc_topk.append(topk_out)

                # Batch encode all top-k texts in one GPU call
                all_topk_texts = [c["text"] for topk in per_doc_topk for c in topk]
                if all_topk_texts:
                    all_embs = self.semantic_chunker.encode_batch(all_topk_texts)
                    offset = 0
                    for topk in per_doc_topk:
                        cnt = len(topk)
                        embs = all_embs[offset: offset + cnt]
                        offset += cnt
                        for chunk, emb in zip(topk, embs):
                            chunk["embedding"] = emb.tolist()

                for (doc_id, doc, acp, _), all_c, topk_c in zip(pending, per_doc_all, per_doc_topk):
                    q_final.put((doc_id, acp, all_c, topk_c))

                pending.clear()
                pending_topk_counts.clear()

            while True:
                item = q_roles.get()
                if item is _STOP:
                    flush()
                    q_final.put(_STOP)
                    return
                doc_id, doc, acp, chunk_dicts = item
                n_chunks = len(chunk_dicts)
                if CHUNKING_CONFIG["top_k"]:
                    topk_n = min(CHUNKING_CONFIG["top_k"], n_chunks)
                else:
                    topk_n = n_chunks
                pending.append((doc_id, doc, acp, chunk_dicts))
                pending_topk_counts.append(topk_n)
                if sum(pending_topk_counts) >= CHUNK_BATCH:
                    flush()

        # ---- Stage 7: streaming upload ----
        UPLOAD_BATCH   = int(os.getenv("UPLOAD_BATCH_SIZE", "128"))
        UPLOAD_WORKERS = int(os.getenv("UPLOAD_WORKERS", "8"))

        upload_stats = {
            "processed": 0,
            "total_chunks": 0,
            "total_top_k": 0,
            "search_uploaded": 0,
            "search_failed": 0,
        }
        _stats_lock = threading.Lock()

        def _flush_upload(batch_docs: List[Tuple]):
            """Upload one batch: ADLS → Search → done markers. Runs in upload pool."""
            if not batch_docs:
                return

            adls_ok: Dict[str, bool] = {}
            def _upload_one(item):
                doc_id, acp, all_chunks, _top_k = item
                return doc_id, self.adls_uploader.upload_json_file(
                    data=all_chunks, adls_path=acp, overwrite=True
                )
            with ThreadPoolExecutor(max_workers=min(len(batch_docs), 4)) as adls_pool:
                for doc_id, ok in adls_pool.map(_upload_one, batch_docs):
                    adls_ok[doc_id] = ok

            if not PIPELINE_CONFIG.get("upload_to_search", True):
                with _stats_lock:
                    for doc_id, acp, all_chunks, _top_k in batch_docs:
                        upload_stats["processed"]    += 1
                        upload_stats["total_chunks"] += len(all_chunks)
                return

            all_topk = [c for _, _, _, top_k in batch_docs for c in top_k]
            if all_topk:
                stats = self.search_uploader.upload_chunks(all_topk, show_progress=False)
                search_failed = stats.get("failed", 0) > 0
                s_uploaded = stats.get("uploaded", 0)
                s_failed   = stats.get("failed", 0)
            else:
                search_failed = False
                s_uploaded = s_failed = 0

            done_paths = []
            with _stats_lock:
                upload_stats["search_uploaded"] += s_uploaded
                upload_stats["search_failed"]   += s_failed
                for doc_id, acp, all_chunks, top_k in batch_docs:
                    upload_stats["processed"]    += 1
                    upload_stats["total_chunks"] += len(all_chunks)
                    upload_stats["total_top_k"]  += len(top_k)
                    if not search_failed and adls_ok.get(doc_id):
                        done_paths.append(acp.replace("_all_chunks.json", f"_done_{self.index_type}.json"))

            if done_paths:
                with ThreadPoolExecutor(max_workers=min(len(done_paths), 4)) as marker_pool:
                    list(marker_pool.map(self.adls_uploader.write_marker, done_paths))
                self._processed_cache.update(done_paths)

        def collector():
            pbar = tqdm(total=total_docs, desc="Processing")
            pending: List[Tuple] = []
            upload_futs = []
            with ThreadPoolExecutor(max_workers=UPLOAD_WORKERS, thread_name_prefix="upload") as pool:
                while True:
                    item = q_final.get()
                    if item is _STOP:
                        if pending:
                            upload_futs.append(pool.submit(_flush_upload, pending))
                        for f in upload_futs:
                            try:
                                f.result()
                            except Exception as e:
                                logger.error(f"Upload error: {e}", exc_info=True)
                        pbar.close()
                        return
                    doc_id, acp, all_chunks, top_k = item
                    pending.append((doc_id, acp, all_chunks, top_k))
                    pbar.update(1)
                    if len(pending) >= UPLOAD_BATCH:
                        upload_futs.append(pool.submit(_flush_upload, pending))
                        pending = []

        # ---- Launch all stages ----
        t_s2 = threading.Thread(target=stage2_producer, name="stage2-split",    daemon=True)
        t_s3 = threading.Thread(target=stage3_encode,   name="stage3-encode",   daemon=True)
        t_s4 = threading.Thread(target=stage4_assemble, name="stage4-assemble", daemon=True)
        t_s5 = threading.Thread(target=stage5_classify, name="stage5-classify", daemon=True)
        t_s6 = threading.Thread(target=stage6_embed,    name="stage6-embed",    daemon=True)
        t_col = threading.Thread(target=collector,       name="collector",       daemon=True)

        for t in (t_s2, t_s3, t_s4, t_s5, t_s6, t_col):
            t.start()
        for t in (t_s2, t_s3, t_s4, t_s5, t_s6, t_col):
            t.join()

        if errors:
            logger.warning(f"Pipeline completed with {len(errors)} errors (see logs above)")

        return upload_stats

    def run(self) -> Dict:
        start = time.time()

        if PIPELINE_CONFIG["create_index"]:
            self.index_manager.create_legal_documents_index(
                index_name=self.index_name,
                doc_type=self.doc_type,
                vector_dimensions=EMBEDDING_CONFIG["dimensions"]
            )

        # Build a set of all already-processed paths once at startup so that
        # _check_one can do O(1) set lookups instead of per-file HTTP HEAD calls.
        logger.info("Building processed-path cache from ADLS (one-time listing)…")
        cache_start = time.time()
        self._processed_cache: set = set(
            self.adls_fetcher.list_files_iter(
                path=self.base_output_path,
                pattern="*.json",
                recursive=True,
            )
        )
        logger.info(
            f"Processed cache built: {len(self._processed_cache):,} paths "
            f"in {time.time() - cache_start:.1f}s"
        )

        # Stream file listing lazily — no full materialisation into memory.
        # Checks done/resume markers and reads JSON only for pending files.
        # Feeds pending docs directly into the GPU pipeline queue.
        pending_q: queue.Queue = queue.Queue(maxsize=512)
        resume_candidates, skipped_ref, stream_thread = self._stream_into_pipeline(pending_q)

        # GPU pipeline consumes from pending_q while streaming is still running.
        # Uploads ADLS + Search inline in rolling batches — no end-of-run bulk upload.
        upload_stats = self._run_queue_pipeline(pending_q)

        stream_thread.join()
        skipped = skipped_ref[0]

        logger.info(
            f"Documents — processed: {upload_stats['processed']}, "
            f"resume: {len(resume_candidates)}, skipped: {skipped}"
        )

        if PIPELINE_CONFIG["upload_to_search"]:
            self._resume_pending_search(resume_candidates)

        stats = {
            "status": "success",
            "doc_type": DOC_TYPE_CONFIG[self.doc_type]["name"],
            "index_type": INDEX_TYPE_CONFIG[self.index_type]["name"],
            "index_name": self.index_name,
            "pipeline_time_seconds": round(time.time() - start, 2),
            "documents_processed": upload_stats["processed"],
            "documents_skipped": skipped,
            "documents_resumed": len(resume_candidates),
            "total_chunks": upload_stats["total_chunks"],
            "top_k_chunks": upload_stats["total_top_k"],
            "search_upload": {
                "uploaded": upload_stats["search_uploaded"],
                "failed":   upload_stats["search_failed"],
            },
        }

        logger.info("PIPELINE COMPLETE")
        return stats


# --------------------------------------------------

def setup_logging():
    log_format = LOGGING_CONFIG["format"]
    log_level = getattr(logging, LOGGING_CONFIG["level"].upper())
    logging.basicConfig(level=log_level, format=log_format,
                        handlers=[logging.StreamHandler()], force=True)


def main():
    parser = argparse.ArgumentParser(description="Production pipeline for legal document indexing")
    parser.add_argument("--doc_type",   type=int, choices=[0, 1], required=True,
                        help="0=High Court  1=Supreme Court")
    parser.add_argument("--index_type", type=int, choices=[0, 1], required=True,
                        help="Index type: 0 = AI Assistant, 1 = Precedent Finder")
    parser.add_argument("--sy", type=int, default=None, help="Start year (inclusive)")
    parser.add_argument("--ey", type=int, default=None, help="End year (inclusive)")
    args = parser.parse_args()

    setup_logging()

    pipeline = ProductionPipeline(
        doc_type=args.doc_type,
        index_type=args.index_type,
        base_output_path="processed",
        start_year=args.sy,
        end_year=args.ey,
    )
    stats = pipeline.run()
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()

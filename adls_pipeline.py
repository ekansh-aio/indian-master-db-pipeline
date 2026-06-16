"""
ADLS-Only Pipeline for Legal Document Processing
Processes documents through chunking + role classification and uploads to ADLS.
Does NOT upload to Azure AI Search. done_0 marker written on successful ADLS upload.

Usage:
    python adls_pipeline.py --doc_type 0 --sy 2010 --ey 2015
        doc_type: 0 = High Court, 1 = Supreme Court

Architecture (6 stages, no Search):
    Stage 1  ThreadPool  IO       Fetch + clean text
    Stage 2  ThreadPool  CPU      Split sentences
    Stage 3  GPU thread  GPU      Batch-encode sentences across docs
    Stage 4  CPU thread  CPU      Assemble SemanticChunk objects
    Stage 5  GPU thread  GPU      Batch role-classify all chunk texts
    Stage 6  ThreadPool  IO       ADLS upload + done_0 marker
"""
import argparse
import json
import logging
import os
import queue
import threading
import time
import hashlib

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
from core.role_classifier import create_classifier_from_config
from utils.weighted_selector import weighted_topk_selection
from tqdm import tqdm

logger = logging.getLogger(__name__)

_STOP = object()

# index_type is always 0 for this pipeline (done_0 marker = ADLS done)
_INDEX_TYPE = 0


# --------------------------------------------------
# UTILITIES  (mirrors production_pipeline.py exactly)
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


def get_done_marker_path(source_file_path: str, base_output: str = "processed") -> str:
    path_without_ext = source_file_path.replace('.json', '')
    parts = Path(path_without_ext).parts
    skip_prefixes = {'raw', 'newapp', 'input', 'data', 'app'}
    relevant_parts = [p for p in parts if p.lower() not in skip_prefixes]
    if relevant_parts:
        directory_parts = relevant_parts[:-1]
        filename = relevant_parts[-1]
        if directory_parts:
            output_dir = "/".join(directory_parts)
            return f"{base_output}/{output_dir}/{filename}_done_0.json"
        return f"{base_output}/{filename}_done_0.json"
    return f"{base_output}/unknown_done_0.json"


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

class ADLSPipeline:

    def __init__(self, doc_type: int, base_output_path: str = "processed",
                 start_year: Optional[int] = None, end_year: Optional[int] = None):
        logger.info("=" * 80)
        logger.info(f"ADLS PIPELINE (no Search) — {DOC_TYPE_CONFIG[doc_type]['name']}")
        logger.info("=" * 80)

        self.doc_type = doc_type
        self.base_output_path = base_output_path
        self.start_year = start_year
        self.end_year = end_year

        if start_year or end_year:
            logger.info(f"Year filter: {start_year or '–'} → {end_year or '–'}")

        self.adls_input_path = DOC_TYPE_CONFIG[doc_type]["adls_input_path"]
        self.role_weights = INDEX_TYPE_CONFIG[_INDEX_TYPE]["role_weights"]

        validate_config()

        self._init_adls()
        self._init_processors()

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

        self.role_classifier = None
        if ROLE_CLASSIFICATION_CONFIG["enabled"]:
            logger.info("Initializing role classifier")
            self.role_classifier = create_classifier_from_config()

    def _stream_into_pipeline(self, pending_q: "queue.Queue") -> Tuple[List[int], threading.Thread]:
        BATCH_SIZE    = int(os.getenv("STREAM_BATCH_SIZE", "30000"))
        CHECK_WORKERS = int(os.getenv("CHECK_WORKERS", "400"))
        io_workers    = PIPELINE_CONFIG.get("io_workers", 64)
        max_docs      = PIPELINE_CONFIG.get("max_documents", None)

        skipped_ref = [0]
        batch_q: "queue.Queue" = queue.Queue(maxsize=2)

        def _check_one(fp: str):
            all_chunks_path = get_all_chunks_adls_path(fp, self.base_output_path)
            done_marker_path = get_done_marker_path(fp, self.base_output_path)
            if done_marker_path in self._processed_cache:
                return "skip", None
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
                f"skipped={skipped_ref[0]}"
            )

        def _lister():
            batch: List[str] = []
            seen = 0

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
                _skip = {'raw', 'newapp', 'input', 'data', 'app'}
                _parts = [p for p in Path(ypath.rstrip('/')).parts
                          if p.lower() not in _skip]
                _proc_year_path = f"{self.base_output_path}/{'/'.join(_parts)}"
                _cache_t0 = time.time()
                try:
                    _year_paths = set(
                        self.adls_fetcher.list_files_iter(
                            path=_proc_year_path,
                            pattern="*.json",
                            recursive=True,
                        )
                    )
                except Exception:
                    _year_paths = set()
                self._processed_cache.update(_year_paths)
                logger.info(
                    f"Lazy cache loaded {len(_year_paths):,} paths for "
                    f"{_proc_year_path!r} in {time.time() - _cache_t0:.1f}s  "
                    f"(total cache size: {len(self._processed_cache):,})"
                )

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
        return skipped_ref, t

    def _run_queue_pipeline(self, pending_docs, total_docs: int = None) -> Dict:
        """
        6-stage producer/consumer pipeline. ADLS upload only — no Search.
        done_0 marker written after successful ADLS upload.
        """
        io_workers       = PIPELINE_CONFIG.get("io_workers", 64)
        assemble_workers = int(os.getenv("ASSEMBLE_WORKERS", "256"))
        SENT_BATCH  = int(os.getenv("SENT_BATCH",  "40000"))
        CHUNK_BATCH = int(os.getenv("CHUNK_BATCH", "20000"))

        q_sentences: queue.Queue = queue.Queue(maxsize=8192)
        q_embedded:  queue.Queue = queue.Queue(maxsize=4096)
        q_chunks:    queue.Queue = queue.Queue(maxsize=4096)
        # Stage 5 → collector: (doc_id, doc, all_chunks_path, chunk_dicts)
        q_roles:     queue.Queue = queue.Queue(maxsize=4096)

        if total_docs is None:
            total_docs = len(pending_docs) if not isinstance(pending_docs, queue.Queue) else None
        errors = []

        # ---- Stage 2: clean + split sentences ----
        def stage2_split(doc: Dict):
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
            with ThreadPoolExecutor(max_workers=io_workers) as pool:
                pending_futs = []
                if isinstance(pending_docs, queue.Queue):
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
            pending_items = []
            pending_counts = []

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

        # ---- Stage 4: CPU chunk assembly ----
        def stage4_assemble():
            def _assemble_one(item):
                doc_id, doc, sentences, sent_embs, cleaned_text, all_chunks_path = item
                chunks, chunk_texts = self.semantic_chunker.assemble_chunks_cpu(
                    cleaned_text, sentences, sent_embs
                )
                if not chunks:
                    return None
                return (doc_id, doc, all_chunks_path, chunks, chunk_texts)

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
            pending: List[Tuple] = []
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

        # ---- Stage 6: ADLS upload + done_0 marker ----
        UPLOAD_BATCH   = int(os.getenv("UPLOAD_BATCH_SIZE", "512"))
        UPLOAD_WORKERS = int(os.getenv("UPLOAD_WORKERS", "64"))

        upload_stats = {
            "processed": 0,
            "total_chunks": 0,
        }
        _stats_lock = threading.Lock()

        def _flush_upload(batch_docs: List[Tuple]):
            """Upload one batch to ADLS and write done_0 markers."""
            if not batch_docs:
                return

            adls_ok: Dict[str, bool] = {}

            def _upload_one(item):
                doc_id, acp, all_chunks = item
                return doc_id, self.adls_uploader.upload_json_file(
                    data=all_chunks, adls_path=acp, overwrite=True
                )

            with ThreadPoolExecutor(max_workers=min(len(batch_docs), 4)) as adls_pool:
                for doc_id, ok in adls_pool.map(_upload_one, batch_docs):
                    adls_ok[doc_id] = ok

            done_paths = []
            with _stats_lock:
                for doc_id, acp, all_chunks in batch_docs:
                    upload_stats["processed"]    += 1
                    upload_stats["total_chunks"] += len(all_chunks)
                    if adls_ok.get(doc_id):
                        done_paths.append(acp.replace("_all_chunks.json", "_done_0.json"))

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
                    item = q_roles.get()
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
                    doc_id, doc, acp, chunk_dicts = item

                    # Build all_chunks with metadata (mirrors stage6 in production_pipeline)
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

                    pending.append((doc_id, acp, all_chunks_out))
                    pbar.update(1)
                    if len(pending) >= UPLOAD_BATCH:
                        upload_futs.append(pool.submit(_flush_upload, pending.copy()))
                        pending.clear()

        t_s2  = threading.Thread(target=stage2_producer, name="stage2-split",    daemon=True)
        t_s3  = threading.Thread(target=stage3_encode,   name="stage3-encode",   daemon=True)
        t_s4  = threading.Thread(target=stage4_assemble, name="stage4-assemble", daemon=True)
        t_s5  = threading.Thread(target=stage5_classify, name="stage5-classify", daemon=True)
        t_col = threading.Thread(target=collector,        name="collector",       daemon=True)

        for t in (t_s2, t_s3, t_s4, t_s5, t_col):
            t.start()
        for t in (t_s2, t_s3, t_s4, t_s5, t_col):
            t.join()

        if errors:
            logger.warning(f"Pipeline completed with {len(errors)} errors (see logs above)")

        return upload_stats

    def run(self) -> Dict:
        start = time.time()

        self._processed_cache: set = set()
        logger.info("Processed-path cache initialised (lazy — loads per year in _lister)")

        pending_q: queue.Queue = queue.Queue(maxsize=2048)
        skipped_ref, stream_thread = self._stream_into_pipeline(pending_q)

        upload_stats = self._run_queue_pipeline(pending_q)

        stream_thread.join()
        skipped = skipped_ref[0]

        elapsed = time.time() - start
        n_proc = upload_stats['processed']
        docs_per_sec = n_proc / elapsed if elapsed > 0 else 0
        logger.info(
            f"Documents — processed: {n_proc}, skipped: {skipped}  |  "
            f"{docs_per_sec:.2f} docs/s  ({elapsed:.1f}s total)"
        )

        stats = {
            "status": "success",
            "doc_type": DOC_TYPE_CONFIG[self.doc_type]["name"],
            "pipeline_time_seconds": round(elapsed, 2),
            "documents_processed": n_proc,
            "documents_skipped": skipped,
            "total_chunks": upload_stats["total_chunks"],
            "docs_per_sec": round(docs_per_sec, 3),
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
    parser = argparse.ArgumentParser(description="ADLS-only pipeline (no Search upload)")
    parser.add_argument("--doc_type", type=int, choices=[0, 1], required=True,
                        help="0=High Court  1=Supreme Court")
    parser.add_argument("--sy", type=int, default=None, help="Start year (inclusive)")
    parser.add_argument("--ey", type=int, default=None, help="End year (inclusive)")
    args = parser.parse_args()

    setup_logging()

    pipeline = ADLSPipeline(
        doc_type=args.doc_type,
        base_output_path="processed",
        start_year=args.sy,
        end_year=args.ey,
    )
    stats = pipeline.run()
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()

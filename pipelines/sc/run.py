"""
SC Judgments Pipeline: ADLS app/ → chunk+classify → ADLS processed/ → top-K → embed → ES.

Single command, end-to-end. Inventory-delta resume built in.

Usage:
    python pipelines/sc/run.py --year-range 2020 2026
    python pipelines/sc/run.py --year-range 2020 2026 --resume --gpu

    # Two-stage (chunk → later embed):
    python pipelines/sc/run.py --stage chunk --year-range 2020 2026
    python pipelines/sc/run.py --stage embed --year-range 2020 2026

    # Force reprocess:
    python pipelines/sc/run.py --year-range 2020 2026 --no-resume

Environment:
    ADLS_ACCOUNT_NAME / ADLS_ACCOUNT_KEY / ADLS_CONTAINER_NAME
    ES_URL / ES_API_KEY (or ES_USER/ES_PASS)
    EMBEDDING_MODEL, EMBEDDING_BATCH_SIZE, IO_WORKERS, ES_BULK_BATCH_SIZE, ...
"""

import argparse
import json
import logging
import os
import queue
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from dotenv import load_dotenv
load_dotenv()

logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(logging.WARNING)
logging.getLogger("azure.storage").setLevel(logging.WARNING)
logging.getLogger("azure.core").setLevel(logging.WARNING)
logging.getLogger("urllib3.connectionpool").setLevel(logging.ERROR)
logging.getLogger("elasticsearch").setLevel(logging.WARNING)

from tqdm import tqdm

from config import (
    LOGGING_CONFIG, ADLS_CONFIG, EMBEDDING_CONFIG, CHUNKING_CONFIG,
    ROLE_CLASSIFICATION_CONFIG, PIPELINE_CONFIG, DOC_TYPE_CONFIG,
    ROLE_WEIGHTS, validate_config,
)
from core.adls_fetcher import ADLSFetcher
from core.adls_uploader import ADLSUploader
from core.semantic_chunker import SemanticChunker
from core.role_classifier import create_classifier_from_config
from core.elasticsearch_client import (
    build_es_client, ensure_index, bulk_mget_exists,
    bulk_upload_with_retry, load_model_multi_gpu, load_embedding_model, embed_texts,
    log_index_stats,
)
from core.progress_tracker import ProgressTracker
from core.inventory import load_inventory, save_inventory, upload_inventory_es, compute_deltas
from core.adls_paths import (
    COLLECTION_CONFIG, get_app_path, get_processed_path,
    get_inventory_path, get_es_inventory_path,
    generate_doc_id, get_all_chunks_path, doc_id_from_path,
)
from utils.weighted_selector import weighted_topk_selection

COLLECTION = "sc"
_STOP = object()

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ES metadata / chunk field lists
# ---------------------------------------------------------------------------

METADATA_FIELDS = [
    "doc_id", "date", "jurisdiction", "doc_name", "year", "bench",
    "court_code", "title", "judge", "pdf_link", "cnr",
    "date_of_registration", "decision_date", "disposal_nature",
    "court", "pdf_exists", "original_source_path", "all_chunks_path",
    "source_file",
]

CHUNK_FIELDS = ["chunk_id", "text", "role", "same_role_chunk_ids"]

TOP_K = 12


# ---------------------------------------------------------------------------
# Helpers (mirror existing adls_pipeline.py — unchanged semantics)
# ---------------------------------------------------------------------------

def attach_same_role_chunk_ids(chunks: List[Dict]) -> None:
    role_to_ids: Dict = defaultdict(list)
    for c in chunks:
        role_to_ids[c.get("role", "Others")].append(c.get("chunk_id") or c.get("id", ""))
    for c in chunks:
        role = c.get("role", "Others")
        c["same_role_chunk_ids"] = [
            cid for cid in role_to_ids[role] if cid != (c.get("chunk_id") or c.get("id", ""))
        ]


def extract_metadata(chunk: Dict) -> Dict:
    meta = {k: chunk[k] for k in METADATA_FIELDS if k in chunk}
    subservice = chunk.get("subservice", {})
    if isinstance(subservice, dict) and "service_type" in subservice:
        meta["service_type"] = subservice["service_type"]
    return meta


# ---------------------------------------------------------------------------
# 7-Stage pipeline (core processing: app/ → processed/ + ES)
# ---------------------------------------------------------------------------

class JudgmentPipeline:
    """End-to-end judgment pipeline for one collection (HC or SC)."""

    def __init__(
        self,
        collection: str,
        base_output_path: str = "processed",
        start_year: Optional[int] = None,
        end_year: Optional[int] = None,
        no_resume: bool = False,
        gpu: bool = True,
        top_k: int = TOP_K,
        stage: str = "all",
    ):
        self.collection = collection
        self.base_output_path = base_output_path
        self.start_year = start_year
        self.end_year = end_year
        self.no_resume = no_resume
        self.gpu = gpu
        self.top_k = top_k
        self.stage = stage

        doc_type_index = 0 if collection == "hc" else 1
        self.doc_type_config = DOC_TYPE_CONFIG[doc_type_index]
        self.app_root = COLLECTION_CONFIG[collection]["app"]
        self.processed_root = COLLECTION_CONFIG[collection]["processed"]
        self.es_index = COLLECTION_CONFIG[collection]["es_index"]
        self.schema_path = COLLECTION_CONFIG[collection]["schema"]

        validate_config()
        self._init_adls()
        self._init_processors()

    def _init_adls(self):
        self.fetcher = ADLSFetcher(
            ADLS_CONFIG["account_name"], ADLS_CONFIG["account_key"],
            ADLS_CONFIG["container_name"],
        )
        self.uploader = ADLSUploader(
            ADLS_CONFIG["account_name"], ADLS_CONFIG["account_key"],
            ADLS_CONFIG["container_name"],
        )

    def _init_processors(self):
        self.semantic_chunker = SemanticChunker(
            model_name=EMBEDDING_CONFIG["model_name"],
            similarity_threshold=CHUNKING_CONFIG["similarity_threshold"],
            min_sentences_per_chunk=CHUNKING_CONFIG["min_sentences_per_chunk"],
            max_sentences_per_chunk=CHUNKING_CONFIG["max_sentences_per_chunk"],
            min_chunk_size=CHUNKING_CONFIG["min_chunk_size"],
        )
        self.role_classifier = None
        if ROLE_CLASSIFICATION_CONFIG["enabled"]:
            log.info("Initializing role classifier")
            self.role_classifier = create_classifier_from_config()

    # ------------------------------------------------------------------
    # Per-year processing
    # ------------------------------------------------------------------

    def process_year(self, year: int) -> Dict:
        log.info("=" * 60)
        log.info("%s — Year %d — starting", self.collection.upper(), year)
        t_start = time.time()

        year_app_path = get_app_path(self.collection, year)
        year_proc_path = get_processed_path(self.collection, year)
        app_inv_path = f"{year_app_path}/_inventory.json"
        proc_inv_path = get_inventory_path(self.collection, year)
        es_inv_path = get_es_inventory_path(self.collection, year)

        # Load inventories
        app_files = load_inventory(self.fetcher, app_inv_path) if not self.no_resume else set()
        if not app_files:
            log.info("  No _inventory.json in app/ — scanning ADLS")
            app_files = self._scan_app_files(year)

        proc_files = load_inventory(self.fetcher, proc_inv_path) if not self.no_resume else set()
        es_files = load_inventory(self.fetcher, es_inv_path) if not self.no_resume else set()

        delta1, delta2 = compute_deltas(app_files, proc_files, es_files, self.collection)

        log.info("  Year %d: app=%d  proc=%d  es=%d  delta1=%d  delta2=%d",
                 year, len(app_files), len(proc_files), len(es_files),
                 len(delta1), len(delta2))

        stats = {"year": year, "processed": 0, "adls_uploaded": 0,
                 "es_uploaded": 0, "es_errors": 0, "skipped_adls": 0, "skipped_es": 0}

        # Delta1: chunk + classify + upload to ADLS processed/
        if self.stage in ("all", "chunk") and delta1:
            chunk_stats = self._run_chunk_pipeline(year, delta1, year_app_path, year_proc_path,
                                                      proc_files, proc_inv_path)
            stats.update(chunk_stats)

        # Delta2: top-K select → embed → ES
        if self.stage in ("all", "embed"):
            es_stats = self._run_es_pipeline(year, delta2, year_proc_path, es_inv_path, es_files)
            stats.update(es_stats)

        elapsed = time.time() - t_start
        stats["elapsed_s"] = round(elapsed, 1)
        log.info("Year %d done — processed=%d  adls=%d  es=%d  errors=%d  %.0fs",
                 year, stats["processed"], stats["adls_uploaded"],
                 stats["es_uploaded"], stats["es_errors"], elapsed)
        return stats

    def _scan_app_files(self, year: int) -> set:
        """Fallback: scan ADLS app/ paths when no inventory exists."""
        base = get_app_path(self.collection, year)
        prefix = base.rstrip("/") + "/"
        files = set()
        for fp in self.fetcher.list_files_iter(path=base, pattern="*.json", recursive=True):
            if fp.endswith("_inventory.json") or "_done_" in fp:
                continue
            rel = fp[len(prefix):] if fp.startswith(prefix) else Path(fp).name
            files.add(rel)
        log.info("  Scanned app/ paths: %d files", len(files))
        return files

    # ------------------------------------------------------------------
    # Chunk pipeline (delta1: app/ → processed/)
    # ------------------------------------------------------------------

    def _run_chunk_pipeline(self, year: int, delta1: set,
                            year_app_path: str, year_proc_path: str,
                            proc_files: set, proc_inv_path: str) -> Dict:
        """6-stage producer/consumer for chunking + ADLS upload (no ES)."""
        log.info("  Chunk pipeline: %d docs to process", len(delta1))

        IO_WORKERS = int(os.getenv("IO_WORKERS", "64"))
        SENT_BATCH = int(os.getenv("SENT_BATCH", "40000"))
        CHUNK_BATCH = int(os.getenv("CHUNK_BATCH", "20000"))
        ASSEMBLE_WORKERS = int(os.getenv("ASSEMBLE_WORKERS", "256"))
        UPLOAD_BATCH = int(os.getenv("UPLOAD_BATCH_SIZE", "512"))
        UPLOAD_WORKERS = int(os.getenv("UPLOAD_WORKERS", "64"))

        q_raw: queue.Queue = queue.Queue()
        q_sentences: queue.Queue = queue.Queue()
        q_embedded: queue.Queue = queue.Queue()
        q_chunks: queue.Queue = queue.Queue()
        q_roles: queue.Queue = queue.Queue()

        errors: List[Exception] = []
        stats_lock = threading.Lock()
        upload_stats = {"processed": 0, "adls_uploaded": 0, "total_chunks": 0}

        # Stream reader: filter by delta1 basenames, read in parallel
        def _read_one(fp: str) -> Optional[Dict]:
            try:
                doc = self.fetcher.read_json_file(fp)
                if doc is not None:
                    doc["_source_file"] = fp
                return doc
            except Exception as e:
                log.warning("Read error %s: %s", fp, e)
                return None

        def _build_path_index():
            """Build relative-path→full_path map for app/ year directory.

            Keyed on relative path from the year root so files in different
            subdirs with the same filename are distinct.
            """
            prefix = year_app_path.rstrip("/") + "/"
            idx = {}
            for fp in self.fetcher.list_files_iter(
                path=year_app_path, pattern="*.json", recursive=True
            ):
                if fp.endswith("_inventory.json") or "_done_" in fp:
                    continue
                rel = fp[len(prefix):] if fp.startswith(prefix) else Path(fp).name
                idx[rel] = fp
            return idx

        def _stream_reader():
            path_idx = _build_path_index()
            batch_buf = []
            for rel_key in sorted(delta1):
                full_path = path_idx.get(rel_key)
                if full_path is None:
                    continue
                batch_buf.append(full_path)
                if len(batch_buf) >= 500:
                    self._read_batch_into_queue(batch_buf, q_raw, errors, _read_one, IO_WORKERS)
                    batch_buf = []
            if batch_buf:
                self._read_batch_into_queue(batch_buf, q_raw, errors, _read_one, IO_WORKERS)
            q_raw.put(_STOP)

        # Stage 2: split sentences
        def stage2_split(doc: Dict):
            source_file = doc.get("_source_file", "unknown.json")
            doc_id = generate_doc_id(source_file, self.collection)
            acp = get_all_chunks_path(source_file, self.collection)
            try:
                text = doc.get("full_text") or doc.get("judgment_text") or doc.get("text", "")
                if not text:
                    return
                sentences = self.semantic_chunker._split_sentences(text)
                if not sentences:
                    return
                q_sentences.put((doc_id, doc, sentences, text, acp))
            except Exception as e:
                log.error("Stage 2 error for %s: %s", doc_id, e)
                errors.append(e)

        def stage2_producer():
            with ThreadPoolExecutor(max_workers=IO_WORKERS) as pool:
                futs = []
                while True:
                    doc = q_raw.get()
                    if doc is _STOP:
                        break
                    futs.append(pool.submit(stage2_split, doc))
                for f in as_completed(futs):
                    exc = f.exception()
                    if exc:
                        errors.append(exc)
            q_sentences.put(_STOP)

        # Stage 3: GPU batch sentence encoding
        def stage3_encode():
            pending = []
            pending_counts = []
            def flush():
                if not pending:
                    return
                all_sents = []
                for _, _, sents, _, _ in pending:
                    all_sents.extend(sents)
                embs = self.semantic_chunker.encode_batch(all_sents)
                offset = 0
                for (did, d, sents, ct, acp), cnt in zip(pending, pending_counts):
                    doc_embs = embs[offset: offset + cnt]
                    offset += cnt
                    q_embedded.put((did, d, sents, doc_embs, ct, acp))
                pending.clear()
                pending_counts.clear()
            while True:
                item = q_sentences.get()
                if item is _STOP:
                    flush()
                    q_embedded.put(_STOP)
                    return
                doc_id, doc, sentences, cleaned, acp = item
                pending.append((doc_id, doc, sentences, cleaned, acp))
                pending_counts.append(len(sentences))
                if sum(pending_counts) >= SENT_BATCH:
                    flush()

        # Stage 4: CPU chunk assembly
        def stage4_assemble():
            def _assemble_one(item):
                did, doc, sents, embs, ct, acp = item
                chunks, chunk_texts = self.semantic_chunker.assemble_chunks_cpu(ct, sents, embs)
                if not chunks:
                    return None
                return (did, doc, acp, chunks, chunk_texts)

            fut_q: queue.Queue = queue.Queue()
            def _submitter():
                with ThreadPoolExecutor(max_workers=ASSEMBLE_WORKERS) as p:
                    while True:
                        item = q_embedded.get()
                        if item is _STOP:
                            break
                        fut_q.put(p.submit(_assemble_one, item))
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
                        errors.append(e)
                        continue
                    if result is not None:
                        q_chunks.put(result)
            t_sub = threading.Thread(target=_submitter, daemon=True)
            t_drain = threading.Thread(target=_drainer, daemon=True)
            t_sub.start(); t_drain.start()
            t_sub.join(); t_drain.join()

        # Stage 5: GPU batch role classification
        def stage5_classify():
            pending = []
            pending_counts = []
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
                        return_probabilities=True,
                    )
                    offset = 0
                    for (did, d, acp, chunks, ctexts), cnt in zip(pending, pending_counts):
                        doc_preds = preds[offset: offset + cnt]
                        offset += cnt
                        enriched = []
                        for ch, pred in zip(chunks, doc_preds):
                            cd = ch.to_dict() if hasattr(ch, "to_dict") else dict(ch)
                            cd["role"] = pred["role"]
                            cd["confidence"] = float(pred["confidence"])
                            enriched.append(cd)
                        q_roles.put((did, d, acp, enriched))
                else:
                    for (did, d, acp, chunks, _) in pending:
                        plain = [(ch.to_dict() if hasattr(ch, "to_dict") else dict(ch)) for ch in chunks]
                        q_roles.put((did, d, acp, plain))
                pending.clear()
                pending_counts.clear()
            while True:
                item = q_chunks.get()
                if item is _STOP:
                    flush()
                    q_roles.put(_STOP)
                    return
                did, doc, acp, chunks, chunk_texts = item
                pending.append((did, doc, acp, chunks, chunk_texts))
                pending_counts.append(len(chunk_texts))
                if sum(pending_counts) >= CHUNK_BATCH:
                    flush()

        # Stage 6: ADLS upload + done marker
        def stage6_adls_upload():
            pending = []
            upload_futs = []
            all_processed_basenames: set = set()

            def _build_all_chunks(doc_id, doc, acp, chunk_dicts):
                source_file = doc.get("_source_file", "unknown.json")
                excluded = {"text", "full_text", "judgment_text", "embedding", "_source_file", "raw_html_text"}
                metadata = {k: v for k, v in doc.items() if k not in excluded}
                if isinstance(metadata.get("metadata"), dict):
                    nested = metadata.pop("metadata")
                    for k, v in nested.items():
                        if k not in {"raw_html", "description"}:
                            metadata[k] = v
                resolved_date = (
                    metadata.get("date") or metadata.get("decision_date", "") or metadata.get("year", "")
                )
                config_jur = self.doc_type_config.get("jurisdiction")
                resolved_jur = config_jur or (
                    metadata.get("jurisdiction") or metadata.get("court", "") or metadata.get("bench", "")
                )
                out = []
                for idx, cd in enumerate(chunk_dicts):
                    out.append({
                        "doc_id": doc_id, "date": resolved_date, "jurisdiction": resolved_jur,
                        **metadata,
                        "id": f"{doc_id}_{idx}", "chunk_id": f"{doc_id}_{idx}",
                        "original_source_path": source_file,
                        "text": cd["text"],
                        "start_char": cd.get("start_char", 0), "end_char": cd.get("end_char", 0),
                        "num_sentences": cd.get("num_sentences", 0),
                        "doc_similarity": float(cd.get("doc_similarity", 0.0)),
                        "avg_similarity": float(cd.get("avg_similarity", 0.0)),
                        "role": cd.get("role", "Others"), "confidence": float(cd.get("confidence", 0.0)),
                    })
                attach_same_role_chunk_ids(out)
                return out

            def _flush_upload(batch) -> set:
                """Upload a batch to ADLS. Returns basenames of successfully uploaded docs."""
                if not batch:
                    return set()
                ok_basenames: set = set()
                def _upload_one(item):
                    did, acp, chunks = item
                    try:
                        ok = self.uploader.upload_json_file(data=chunks, adls_path=acp, overwrite=True)
                    except Exception as e:
                        log.warning("ADLS upload failed for %s: %s", acp, e)
                        ok = False
                    return did, Path(acp).name, ok, len(chunks)
                with ThreadPoolExecutor(max_workers=min(len(batch), 4)) as adls_pool:
                    results = list(adls_pool.map(_upload_one, batch))
                with stats_lock:
                    for did, basename, ok, n_chunks in results:
                        if ok:
                            upload_stats["adls_uploaded"] += 1
                            upload_stats["total_chunks"] += n_chunks
                            ok_basenames.add(basename)
                return ok_basenames

            pbar = tqdm(desc=f"Year {year} (chunk)", unit="doc")
            with ThreadPoolExecutor(max_workers=UPLOAD_WORKERS) as pool:
                while True:
                    item = q_roles.get()
                    if item is _STOP:
                        if pending:
                            upload_futs.append(pool.submit(_flush_upload, list(pending)))
                        for f in upload_futs:
                            try:
                                ok_set = f.result()
                                all_processed_basenames.update(ok_set)
                            except Exception as e:
                                log.error("Upload error: %s", e)
                        pbar.close()
                        # Save updated processed inventory (only successfully uploaded docs)
                        if all_processed_basenames:
                            all_proc = proc_files | all_processed_basenames if not self.no_resume else all_processed_basenames
                            save_inventory(self.uploader, proc_inv_path, all_proc,
                                           self.es_index, year, year_proc_path)
                        return
                    did, doc, acp, chunk_dicts = item
                    all_chunks = _build_all_chunks(did, doc, acp, chunk_dicts)
                    pending.append((did, acp, all_chunks))
                    with stats_lock:
                        upload_stats["processed"] += 1
                    pbar.update(1)
                    if len(pending) >= UPLOAD_BATCH:
                        upload_futs.append(pool.submit(_flush_upload, list(pending)))
                        pending.clear()

        threads = [
            threading.Thread(target=_stream_reader, name="reader", daemon=True),
            threading.Thread(target=stage2_producer, name="s2-split", daemon=True),
            threading.Thread(target=stage3_encode, name="s3-encode", daemon=True),
            threading.Thread(target=stage4_assemble, name="s4-assemble", daemon=True),
            threading.Thread(target=stage5_classify, name="s5-classify", daemon=True),
            threading.Thread(target=stage6_adls_upload, name="s6-adls", daemon=True),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        if errors:
            log.warning("Chunk pipeline completed with %d errors", len(errors))

        return upload_stats

    def _read_batch_into_queue(self, paths, q, errors, reader_fn, workers):
        def _worker(fp):
            doc = reader_fn(fp)
            if doc:
                q.put(doc)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(_worker, paths))

    # ------------------------------------------------------------------
    # ES pipeline (delta2: processed/ → ES)
    # ------------------------------------------------------------------

    def _run_es_pipeline(self, year: int, delta2: set,
                         year_proc_path: str, es_inv_path: str,
                         es_files: set) -> Dict:
        log.info("  ES pipeline: %d docs to upload", len(delta2))

        if not delta2:
            return {"es_uploaded": 0, "es_errors": 0, "skipped_es": 0}

        es = build_es_client()
        schema_path = self.schema_path
        ensure_index(es, self.es_index, schema_path, recreate=False)

        if self.gpu:
            model, pool = load_model_multi_gpu()
        else:
            model, pool = load_embedding_model(device="cpu"), None

        tracker = ProgressTracker(self.collection, f"year={year}",
                                  no_resume=self.no_resume)

        IO_WORKERS = int(os.getenv("IO_WORKERS", "32"))
        BUFFER_SIZE = int(os.getenv("PIPELINE_BUFFER_SIZE", "500"))

        stats = {"es_uploaded": 0, "es_errors": 0, "skipped_es": 0, "scanned": 0}

        # Build full-path index from processed/ paths
        proc_path_idx: Dict[str, str] = {}
        for fp in self.fetcher.list_files_iter(
            path=year_proc_path, pattern="*_all_chunks.json", recursive=True
        ):
            proc_path_idx[Path(fp).name] = fp

        path_buf: List[str] = []
        all_new_ids: set = set()

        for basename in sorted(delta2):
            full_path = proc_path_idx.get(basename)
            if full_path is None:
                continue
            path_buf.append(full_path)
            stats["scanned"] += 1
            if len(path_buf) >= BUFFER_SIZE:
                self._flush_es_batch(path_buf, es, model, pool, tracker, stats, all_new_ids)
                path_buf = []

        if path_buf:
            self._flush_es_batch(path_buf, es, model, pool, tracker, stats, all_new_ids)

        # Write ES inventory
        merged_es = es_files | all_new_ids
        upload_inventory_es(self.fetcher, self.collection, year, merged_es)

        if pool is not None:
            model.stop_multi_process_pool(pool)

        return stats

    def _flush_es_batch(self, paths, es, model, pool, tracker, stats, all_new_ids):
        """Read a batch of processed/ files, top-K select, embed, upload to ES."""
        raw = self._read_batch_parallel(paths)

        # ES mget pre-filter
        if not self.no_resume:
            path_ids = {p: doc_id_from_path(p) for p, _ in raw}
            found = bulk_mget_exists(es, self.es_index, list(path_ids.values()))
            stats["skipped_es"] += len(found)
            raw = [(p, c) for p, c in raw if path_ids[p] not in found]
            if not raw:
                return

        # Prepare docs: top-K selection
        prepared = []
        all_texts = []
        offsets = []

        for path, chunks in raw:
            try:
                indices = weighted_topk_selection(
                    chunks, top_k=self.top_k,
                    similarity_key="doc_similarity", role_weights=ROLE_WEIGHTS,
                )
                selected = [chunks[i] for i in indices]
                metadata = extract_metadata(chunks[0])
                if "all_chunks_path" not in metadata:
                    metadata["all_chunks_path"] = path
                slim = [{k: c[k] for k in CHUNK_FIELDS if k in c} for c in selected]
                texts = [c.get("text", "") for c in selected]
                offsets.append(len(all_texts))
                all_texts.extend(texts)
                prepared.append((metadata, slim))
            except Exception as e:
                log.warning("  Skipping %s: %s", path, e)
                offsets.append(None)
                prepared.append(None)

        if not all_texts:
            return

        # GPU embed
        embeddings = embed_texts(model, all_texts, pool)

        # Attach embeddings, build parent docs, upload
        parent_docs = []
        for i, item in enumerate(prepared):
            if item is None or offsets[i] is None:
                continue
            metadata, slim_chunks = item
            offset = offsets[i]
            for j, sc in enumerate(slim_chunks):
                sc["embedding"] = embeddings[offset + j].tolist()
            doc = dict(metadata)
            for bool_field in ("pdf_exists",):
                if bool_field in doc and isinstance(doc[bool_field], str):
                    doc[bool_field] = doc[bool_field].strip().lower() == "true"
            doc["chunks"] = slim_chunks
            parent_docs.append(doc)

        if not parent_docs:
            return

        ok, errs = bulk_upload_with_retry(es, self.es_index, parent_docs)
        stats["es_uploaded"] += ok
        stats["es_errors"] += errs

        new_ids = {d.get("doc_id", "") for d in parent_docs if d.get("doc_id")}
        tracker.mark_batch_done(new_ids)
        all_new_ids.update(new_ids)

    def _read_batch_parallel(self, paths, workers: int = None):
        """Read a batch of processed/ JSON files in parallel."""
        if workers is None:
            workers = int(os.getenv("IO_WORKERS", "32"))
        results: List[Tuple[str, List[Dict]]] = []
        lock = threading.Lock()
        sem = threading.Semaphore(workers)

        def _worker(path):
            nonlocal results
            try:
                data = self.fetcher.read_json_file(path)
                if isinstance(data, list) and data:
                    with lock:
                        results.append((path, data))
            except Exception as e:
                log.warning("Read error %s: %s", path, e)
            finally:
                sem.release()

        threads = []
        for p in paths:
            sem.acquire()
            t = threading.Thread(target=_worker, args=(p,), daemon=True)
            t.start()
            threads.append(t)
        for t in threads:
            t.join()
        return results

    # ------------------------------------------------------------------
    # Main runner
    # ------------------------------------------------------------------

    def run(self) -> Dict:
        log.info("=" * 80)
        log.info("%s PIPELINE — %s", self.collection.upper(), self.doc_type_config["name"])
        log.info("Stage: %s  |  GPU: %s  |  Resume: %s", self.stage, self.gpu, not self.no_resume)
        log.info("=" * 80)

        years = self._resolve_years()
        all_stats = []
        total_start = time.time()

        for i, year in enumerate(years, 1):
            log.info("[%d/%d] Processing year %d", i, len(years), year)
            s = self.process_year(year)
            all_stats.append(s)

        total_elapsed = time.time() - total_start
        total_processed = sum(s.get("processed", 0) for s in all_stats)
        total_adls = sum(s.get("adls_uploaded", 0) for s in all_stats)
        total_es = sum(s.get("es_uploaded", 0) for s in all_stats)
        total_errors = sum(s.get("es_errors", 0) for s in all_stats)

        log.info("=" * 60)
        log.info("%s PIPELINE COMPLETE", self.collection.upper())
        log.info("Years : %s", years)
        log.info("Processed (chunked) : %s", f"{total_processed:,}")
        log.info("ADLS uploaded       : %s", f"{total_adls:,}")
        log.info("ES uploaded         : %s", f"{total_es:,}")
        log.info("ES errors           : %s", f"{total_errors:,}")
        log.info("Wall time           : %.0f s (%.1f min)", total_elapsed, total_elapsed / 60)

        for s in all_stats:
            log.info("  year=%s  processed=%s  adls=%s  es=%s  errors=%s  %.0fs",
                     s["year"],
                     f"{s.get('processed', 0):,}",
                     f"{s.get('adls_uploaded', 0):,}",
                     f"{s.get('es_uploaded', 0):,}",
                     f"{s.get('es_errors', 0):,}",
                     s.get("elapsed_s", 0))

        if total_es > 0:
            es = build_es_client()
            log_index_stats(es, self.es_index)

        return {"status": "success", "years": years, "stats": all_stats}

    def _resolve_years(self) -> List[int]:
        sy = self.start_year if self.start_year is not None else 1950
        ey = self.end_year   if self.end_year   is not None else 2026
        return list(range(sy, ey + 1))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def setup_logging():
    fmt = "%(asctime)s %(levelname)s [%(name)s] %(message)s"
    level = getattr(logging, LOGGING_CONFIG["level"].upper(), logging.INFO)
    logging.basicConfig(level=level, format=fmt,
                        handlers=[logging.StreamHandler()], force=True)


def main():
    ap = argparse.ArgumentParser(description="SC Judgments: app/ → chunk → processed/ → embed → ES")
    ap.add_argument("--year-range", nargs=2, type=int, default=None,
                    metavar=("FROM", "TO"), help="Inclusive year range")
    ap.add_argument("--years", nargs="+", type=int, default=None,
                    metavar="YEAR", help="Specific years")
    ap.add_argument("--stage", choices=["all", "chunk", "embed"], default="all",
                    help="Pipeline stage to run (default: all)")
    ap.add_argument("--no-resume", action="store_true",
                    help="Disable resume — reprocess everything")
    ap.add_argument("--no-gpu", action="store_true", dest="no_gpu",
                    help="Disable GPU (use CPU)")
    ap.add_argument("--top-k", type=int, default=TOP_K,
                    help=f"Chunks per doc for ES (default: {TOP_K})")
    args = ap.parse_args()

    setup_logging()

    if args.year_range and args.years:
        log.error("Use --years or --year-range, not both")
        sys.exit(1)

    if args.year_range:
        a, b = args.year_range
        target_years = list(range(min(a, b), max(a, b) + 1))
    elif args.years:
        target_years = args.years
    else:
        target_years = list(range(2020, 2027))

    pipeline = JudgmentPipeline(
        collection=COLLECTION,
        start_year=min(target_years),
        end_year=max(target_years),
        no_resume=args.no_resume,
        gpu=not args.no_gpu,
        top_k=args.top_k,
        stage=args.stage,
    )
    pipeline.run()


if __name__ == "__main__":
    main()

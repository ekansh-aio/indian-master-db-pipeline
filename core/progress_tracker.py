"""
Progress tracking for resumable pipelines.

Each pipeline has a local JSONL progress file at:
    pipeline_progress/done_ids_{name}_{partition_key}.jsonl

Where name is e.g. "hc", "central_acts", "state_acts" and partition_key is
e.g. "year=2020" or "state=karnataka".

The file is append-only (O(1) per flush). On startup the tracker loads all
previously recorded IDs into a set for O(1) skip checks.
"""

import json
import logging
import os
from pathlib import Path
from typing import Optional, Set

log = logging.getLogger(__name__)

PROGRESS_DIR = Path("pipeline_progress")


class ProgressTracker:
    """Tracks which doc_ids have been successfully processed for a partition.

    Usage:
        tracker = ProgressTracker("hc", "year=2020")
        if tracker.is_done(doc_id):
            continue  # skip
        # ... process doc ...
        tracker.mark_done(doc_id)

        # Or batch:
        tracker.mark_batch_done([doc_id_1, doc_id_2, ...])
    """

    def __init__(
        self,
        name: str,
        partition_key: str,
        no_resume: bool = False,
        progress_dir: Optional[Path] = None,
    ):
        """
        Args:
            name: Pipeline/collection name, e.g. "hc", "state_acts".
            partition_key: e.g. "year=2020", "state=karnataka".
            no_resume: If True, start fresh (ignore existing progress).
            progress_dir: Override the default progress directory.
        """
        self.name = name
        self.partition_key = partition_key
        self._dir = (progress_dir or PROGRESS_DIR)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._path = self._dir / f"done_ids_{name}_{partition_key}.jsonl"

        self._done: Set[str] = set()
        if not no_resume:
            self._load()

    def _load(self) -> None:
        if not self._path.exists():
            return
        count = 0
        for line in self._path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    self._done.add(json.loads(line))
                except json.JSONDecodeError:
                    self._done.add(line.strip('"'))
                count += 1
        if count:
            log.info("  ProgressTracker [%s %s]: loaded %d done IDs", self.name, self.partition_key, count)

    def is_done(self, doc_id: str) -> bool:
        return doc_id in self._done

    def mark_done(self, doc_id: str) -> None:
        if doc_id in self._done:
            return
        self._done.add(doc_id)
        with self._path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(doc_id) + "\n")

    def mark_batch_done(self, doc_ids: Set[str]) -> None:
        if not doc_ids:
            return
        new_ids = doc_ids - self._done
        if not new_ids:
            return
        self._done.update(new_ids)
        with self._path.open("a", encoding="utf-8") as f:
            for doc_id in new_ids:
                f.write(json.dumps(doc_id) + "\n")

    @property
    def done_ids(self) -> Set[str]:
        return self._done

    @property
    def count(self) -> int:
        return len(self._done)

"""
JSON serialisation utilities for pipeline data.

Changes vs original (v2):
    FIX-1  _sanitize: added handling for tuples (treated as lists) — pipeline
           stages occasionally produce tuple values that json.dump cannot handle.
    FIX-2  EnhancedJSONEncoder.default: added list handling so floats inside
           plain Python lists that contain NaN/Inf are also sanitised. Without
           this, json.dumps({"v": [float("nan")]}) would still produce invalid JSON
           even when using the encoder.
    FIX-3  EnhancedJSONEncoder.default: added bytes handling (base64 decode to str)
           so accidental bytes values don't crash the serialiser silently.
    FIX-4  safe_json_dump / safe_json_dumps: added an explicit ensure_ascii=False
           flag (was already in dumps but missing in the mental model — unified).
"""
import json
import math
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Any


def _sanitize(obj: Any) -> Any:
    """
    Recursively sanitize data for JSON serialization.
      - NaN / Inf floats       → 0.0
      - numpy scalars          → Python native types
      - numpy arrays           → lists
      - tuples                 → lists          (FIX-1)
      - Path / datetime        → strings
      - bytes                  → utf-8 string   (FIX-3)
    """
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):          # FIX-1: handle tuples
        return [_sanitize(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return _sanitize(obj.tolist())            # recurse so NaN inside arrays is caught
    elif isinstance(obj, np.generic):
        val = obj.item()
        if isinstance(val, float) and not math.isfinite(val):
            return 0.0
        return val
    elif isinstance(obj, float) and not math.isfinite(obj):
        return 0.0
    elif isinstance(obj, bytes):                  # FIX-3
        return obj.decode("utf-8", errors="replace")
    elif isinstance(obj, Path):
        return str(obj)
    elif isinstance(obj, datetime):
        return obj.isoformat()
    return obj


class EnhancedJSONEncoder(json.JSONEncoder):
    """
    Drop-in JSONEncoder that handles numpy types, NaN/Inf, Path, datetime, bytes.
    Use as: json.dumps(data, cls=EnhancedJSONEncoder)
    Note: for pipeline bulk serialisation prefer safe_json_dump() which pre-sanitises
    the whole structure once via _sanitize() rather than per-object dispatch.
    """

    def default(self, o: Any) -> Any:
        if isinstance(o, np.generic):
            val = o.item()
            if isinstance(val, float) and not math.isfinite(val):
                return 0.0
            return val
        if isinstance(o, np.ndarray):
            return _sanitize(o.tolist())          # FIX-2: recurse so inner NaN is caught
        if isinstance(o, bytes):                  # FIX-3
            return o.decode("utf-8", errors="replace")
        if isinstance(o, Path):
            return str(o)
        if isinstance(o, datetime):
            return o.isoformat()
        return super().default(o)

    # FIX-2: override iterencode so plain Python lists containing NaN/Inf are
    # sanitised before the C encoder touches them (default() is not called for
    # built-in types like float, even inside lists).
    def encode(self, o: Any) -> str:
        return super().encode(_sanitize(o))

    def iterencode(self, o: Any, _one_shot: bool = False):
        return super().iterencode(_sanitize(o), _one_shot=_one_shot)


def safe_json_dump(data: Any, file_path, indent: int = 2) -> None:
    """Dump pipeline data safely to a JSON file. Handles numpy, NaN, Inf, datetime, Path, bytes."""
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(_sanitize(data), f, indent=indent, ensure_ascii=False)


def safe_json_dumps(data: Any, indent: int = 2) -> str:
    """Serialize pipeline data to a JSON string. Handles numpy, NaN, Inf, datetime, Path, bytes."""
    return json.dumps(_sanitize(data), indent=indent, ensure_ascii=False)

import json
import numpy as np
from pathlib import Path
from datetime import datetime


class EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, o):
        # numpy scalars (float32, int64, etc)
        if isinstance(o, np.generic):
            return o.item()

        # numpy arrays
        if isinstance(o, np.ndarray):
            return o.tolist()

        # pathlib
        if isinstance(o, Path):
            return str(o)

        # datetime
        if isinstance(o, datetime):
            return o.isoformat()

        return super().default(o)


def safe_json_dump(data, file_path, indent=2):
    """
    Dump any pipeline object safely to JSON.
    Handles numpy, float32, arrays, datetime, etc.
    """
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            indent=indent,
            cls=EnhancedJSONEncoder,
            ensure_ascii=False
        )


def safe_json_dumps(data, indent=2):
    """
    JSON string version (useful for logs or printing)
    """
    return json.dumps(
        data,
        indent=indent,
        cls=EnhancedJSONEncoder,
        ensure_ascii=False
    )

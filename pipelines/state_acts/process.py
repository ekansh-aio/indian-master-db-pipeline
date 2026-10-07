"""
State Acts Processor — PDF → Surya OCR → IndicTrans2 translation → semantic chunks → embeddings → ADLS.

ADLS paths use the new convention: app/State_Acts/, processed/State_Acts/ (not state_acts/app/).

Callable standalone or via run.py:
    python pipelines/state_acts/process.py --states karnataka,delhi,maharashtra,haryana
    python pipelines/state_acts/process.py --states maharashtra --no-upload
    python pipelines/state_acts/process.py --all-states
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# Reuse all processing logic from the old module
import pipeline.process_state_acts as _old

# ---------------------------------------------------------------------------
# Patch ADLS paths in process_act:
#   Old: state_acts/app/state={s}/year={y}/{id}.json
#   Old: state_acts/processed/state={s}/year={y}/{id}.json
#   New: app/State_Acts/state={s}/year={y}/{id}.json
#   New: processed/State_Acts/state={s}/year={y}/{id}.json
#
# The old module hardcodes these inline strings. We monkey-patch the upload
# client to remap them at call time.
# ---------------------------------------------------------------------------

_OLD_PREFIXES = ("state_acts/app/", "state_acts/processed/")
_NEW_PREFIXES = ("app/State_Acts/", "processed/State_Acts/")


def _make_path_patching_uploader(inner):
    class _Patcher:
        def upload_json_file(self, data, adls_path, overwrite=True):
            for old_pre, new_pre in zip(_OLD_PREFIXES, _NEW_PREFIXES):
                if adls_path.startswith(old_pre):
                    adls_path = new_pre + adls_path[len(old_pre):]
                    break
            return inner.upload_json_file(data, adls_path, overwrite)

        def __getattr__(self, name):
            return getattr(inner, name)

    return _Patcher()


_original_process_act = _old.process_act


def _patched_process_act(meta, pdf_dir, state_key, ctx):
    original_uploader = ctx.uploader
    ctx.uploader = _make_path_patching_uploader(original_uploader)
    try:
        return _original_process_act(meta, pdf_dir, state_key, ctx)
    finally:
        ctx.uploader = original_uploader


_old.process_act = _patched_process_act

if __name__ == "__main__":
    _old.main()

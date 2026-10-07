# utils/

Small shared utilities.

| File | Purpose |
|---|---|
| `weighted_selector.py` | Selects top-K chunks from a document using a weighted combination of role importance and doc similarity score. Used by HC and SC ES stage to pick the best 12 chunks per document. |
| `json_helper.py` | Safe JSON serialisation that handles numpy arrays and NaN/Infinity values, which appear in embedding outputs. |

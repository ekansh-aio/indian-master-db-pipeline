# schemas/

Elasticsearch index mapping definitions. One file per collection.

| File | Index | Shards | Notes |
|---|---|---|---|
| `hc.json` | `hc_judgements` | 8 | Nested chunks with 384-dim dense vector; rich court metadata (bench, court_code, CNR, etc.) |
| `sc.json` | `sc_judgements` | 8 | Identical structure to `hc.json` |
| `central_acts.json` | `central_acts` | 2 | Nested chunks; act metadata (act_name, act_number, handle_id) |
| `state_acts.json` | `state_acts` | 4 | Nested chunks; extra fields for OCR (ocr_applied, ocr_page_count), translation (_is_translated, language_original) |
| `legal_dictionary.json` | `legal_dictionary` | — | Flat doc: term, part_of_speech, definition, embedding[384] |

These are applied once at index creation via `ensure_index()` in `core/elasticsearch_client.py`.
To recreate an index from scratch, pass `--recreate-index` to the relevant pipeline's `--stage es`.

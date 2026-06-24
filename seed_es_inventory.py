"""
Two jobs in one pass:
  1. For every year in processed/High_Court_Judgements/year=Y/_inventory.json on ADLS,
     upload a copy as inventory_es.json at the same path (marks current ADLS state as
     "all uploaded to ES").
  2. Seed pipeline_progress/done_ids_hc_year=Y.jsonl with all doc_id stems from that
     inventory, so adls_to_es_pipeline.py --local-resume skips them.

Safe to re-run: ADLS upload is overwrite=True; local JSONL only written if count in
file is less than inventory count (avoids duplicating lines on re-run).

Usage:
    cd /home/azureuser/masdb
    source .env   (or export vars)
    .venv/bin/python pipeline/seed_es_inventory.py
"""
import json
import os
import sys
from pathlib import Path

from azure.storage.filedatalake import DataLakeServiceClient

PROGRESS_DIR = Path("/home/azureuser/masdb/pipeline_progress")
PROCESSED_ROOT = "processed/High_Court_Judgements"

def main():
    account   = os.environ["ADLS_ACCOUNT_NAME"]
    key       = os.environ["ADLS_ACCOUNT_KEY"]
    container = os.environ["ADLS_CONTAINER_NAME"]

    client = DataLakeServiceClient(
        account_url=f"https://{account}.dfs.core.windows.net",
        credential=key)
    fs = client.get_file_system_client(container)

    PROGRESS_DIR.mkdir(exist_ok=True)

    # Discover all years that have a _inventory.json
    print("Scanning processed/ for _inventory.json files...")
    inv_paths = []
    for item in fs.get_paths(PROCESSED_ROOT):
        if not item.is_directory and item.name.endswith("/_inventory.json") and "year=" in item.name:
            inv_paths.append(item.name)
    inv_paths.sort()
    print(f"Found {len(inv_paths)} inventory files")

    for adls_path in inv_paths:
        # Extract year
        year_str = adls_path.split("year=")[-1].split("/")[0]
        try:
            year = int(year_str)
        except ValueError:
            print(f"  SKIP: cannot parse year from {adls_path}")
            continue

        # Download inventory
        try:
            fc = fs.get_file_client(adls_path)
            data = json.loads(fc.download_file().readall())
        except Exception as e:
            print(f"  [{year}] ERROR downloading inventory: {e}")
            continue

        files = data.get("files", [])
        file_count = data.get("file_count", len(files))

        # ── Job 1: upload inventory_es.json ──────────────────────────────────
        es_inv_path = adls_path.replace("/_inventory.json", "/inventory_es.json")
        es_inv_data = dict(data)
        es_inv_data["es_uploaded"] = True
        es_inv_data["es_seeded_from"] = adls_path
        try:
            es_fc = fs.get_file_client(es_inv_path)
            payload = json.dumps(es_inv_data, ensure_ascii=False).encode("utf-8")
            es_fc.upload_data(payload, overwrite=True)
            print(f"  [{year}] inventory_es.json uploaded ({file_count:,} docs)")
        except Exception as e:
            print(f"  [{year}] ERROR uploading inventory_es.json: {e}")

        # ── Job 2: seed done_ids JSONL ────────────────────────────────────────
        jsonl_path = PROGRESS_DIR / f"done_ids_hc_year={year}.jsonl"

        # Count existing lines to avoid duplication
        existing_count = 0
        if jsonl_path.exists():
            existing_count = sum(1 for line in jsonl_path.read_text(encoding="utf-8").splitlines() if line.strip())

        if existing_count >= file_count:
            print(f"  [{year}] done_ids already has {existing_count:,} lines >= {file_count:,} — skip")
            continue

        # Write all doc_id stems (filename minus _all_chunks.json)
        doc_ids = [f.replace("_all_chunks.json", "") for f in files]

        # Write fresh (overwrite) since we want exactly the inventory contents
        with open(jsonl_path, "w", encoding="utf-8") as fh:
            for doc_id in doc_ids:
                fh.write(json.dumps(doc_id) + "\n")
        print(f"  [{year}] done_ids seeded: {len(doc_ids):,} entries -> {jsonl_path.name}")

    print("\nDone.")

if __name__ == "__main__":
    main()

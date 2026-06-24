"""
Query ES for document counts per year in hc_judgements.
Run: .venv/bin/python pipeline/verify_hc_years.py
"""
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv
from elasticsearch import Elasticsearch

load_dotenv()

es = Elasticsearch(
    os.environ["ES_URL"],
    api_key=os.environ["ES_API_KEY"],
    request_timeout=60,
)

resp = es.search(
    index="hc_judgements",
    body={
        "size": 0,
        "aggs": {
            "by_year": {
                "terms": {
                    "field": "year",
                    "size": 100,
                    "order": {"_key": "asc"},
                }
            }
        },
    },
)

total = resp["hits"]["total"]["value"]
buckets = resp["aggregations"]["by_year"]["buckets"]

print(f"{'Year':>6}  {'Docs':>12}")
print("-" * 22)
for b in buckets:
    print(f"{b['key']:>6}  {b['doc_count']:>12,}")

print("-" * 22)
print(f"{'Total':>6}  {total:>12,}")
print(f"\n{len(buckets)} years in index")

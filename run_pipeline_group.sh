#!/usr/bin/env bash
# run_pipeline_group.sh <GROUP_NAME> <YEAR1> [YEAR2 ...]
#
# For each year in the group, runs the full pipeline in sequence:
#   Stage A: hc_upload/pipeline.py --year Y   (S3 -> ADLS app/ + processed/)
#   Stage B: adls_to_es_pipeline.py --years Y (ADLS processed/ -> ES)
#
# Respects inventories at every stage:
#   - pipeline.py reads processed/_inventory.json to skip already-uploaded files (selective download)
#   - adls_to_es uses --local-resume to skip doc_ids already in done_ids JSONL
#
# Usage:
#   bash run_pipeline_group.sh recent 2024 2025 2026
#   bash run_pipeline_group.sh old    1950 1951 1952 ...

set -euo pipefail

GROUP="${1:-group}"
shift
YEARS=("$@")

if [ ${#YEARS[@]} -eq 0 ]; then
    echo "Usage: $0 <group_name> <year1> [year2 ...]"
    exit 1
fi

cd /home/azureuser/masdb
source .env
export $(grep -v "^#" .env | xargs)

VENV="/home/azureuser/masdb/.venv/bin/python"
LOG_DIR="/home/azureuser/masdb/logs"
mkdir -p "$LOG_DIR"

echo "============================================================"
echo "Pipeline group: $GROUP"
echo "Years: ${YEARS[*]}"
echo "Started: $(date)"
echo "============================================================"

for YEAR in "${YEARS[@]}"; do
    echo ""
    echo "──────────────────────────────────────────────────────────"
    echo "YEAR $YEAR — Stage A: hc_upload (S3 -> ADLS app/ + processed/)"
    echo "──────────────────────────────────────────────────────────"
    LOG_A="$LOG_DIR/upload_${GROUP}_${YEAR}.log"

    cd /home/azureuser/masdb/hc_upload
    "$VENV" pipeline.py --year "$YEAR" 2>&1 | tee "$LOG_A"
    RC=$?
    if [ $RC -ne 0 ]; then
        echo "ERROR: hc_upload pipeline.py failed for year $YEAR (exit $RC)" | tee -a "$LOG_A"
        echo "Skipping ES stage for year $YEAR, continuing to next year..."
        continue
    fi
    echo "Stage A done for year $YEAR"

    echo ""
    echo "──────────────────────────────────────────────────────────"
    echo "YEAR $YEAR — Stage B: adls_to_es (ADLS processed/ -> ES)"
    echo "──────────────────────────────────────────────────────────"
    LOG_B="$LOG_DIR/es_${GROUP}_${YEAR}.log"

    cd /home/azureuser/masdb
    "$VENV" pipeline/adls_to_es_pipeline.py --years "$YEAR" 2>&1 | tee "$LOG_B"
    RC=$?
    if [ $RC -ne 0 ]; then
        echo "ERROR: adls_to_es_pipeline.py failed for year $YEAR (exit $RC)" | tee -a "$LOG_B"
        echo "Continuing to next year..."
    else
        echo "Stage B done for year $YEAR"
    fi
done

echo ""
echo "============================================================"
echo "Group $GROUP COMPLETE"
echo "Years processed: ${YEARS[*]}"
echo "Finished: $(date)"
echo "============================================================"

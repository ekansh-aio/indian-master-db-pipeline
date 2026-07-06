#!/usr/bin/env bash
# launch_overnight.sh
#
# 1. Fixes hc_upload/pipeline.py parquet bug
# 2. Seeds pipeline_progress/done_ids_hc_year=Y.jsonl + uploads inventory_es.json
#    on ADLS for all years already in processed/ (marks them as ES-done)
# 3. Launches tmux sessions for 4 year groups running in parallel:
#
#    group_recent  : 2026 2025 2024 2023 2022 2021 2020      (newest first - high value)
#    group_mid     : 2019 2018 2017 2016 2015 2014 2013 2012
#    group_old1    : 2011 2010 2009 2008 2007 2006 2005 2004
#    group_old2    : 2003 2002 2001 2000 1999 ... 1950        (pre-2004)
#
# Each session runs run_pipeline_group.sh which chains:
#   hc_upload/pipeline.py --year Y  ->  adls_to_es_pipeline.py --years Y --local-resume
# for each year in sequence within that group.
#
# Sessions survive terminal disconnect. Check with:
#   tmux ls
#   tmux attach -t group_recent

set -euo pipefail
cd /home/azureuser/masdb
source .env
export $(grep -v "^#" .env | xargs)

VENV=".venv/bin/python"
SCRIPT="/home/azureuser/masdb/run_pipeline_group.sh"
LOG_DIR="/home/azureuser/masdb/logs"
mkdir -p "$LOG_DIR"

echo "=== Step 1: Fix parquet bug in hc_upload/pipeline.py ==="
$VENV /home/azureuser/masdb/pipeline/fix_parquet.py
echo ""

echo "=== Step 2: Seed ES inventory (ADLS inventory_es.json + local done_ids) ==="
$VENV /home/azureuser/masdb/pipeline/seed_es_inventory.py
echo ""

echo "=== Step 3: Launch tmux pipeline groups ==="

# Kill any leftover sessions from prior runs
for sess in group_recent group_mid group_old1 group_old2; do
    tmux kill-session -t "$sess" 2>/dev/null && echo "Killed old tmux session: $sess" || true
done

# Year groups — recent first so high-value years complete first
declare -A GROUPS
GROUPS["group_recent"]="2026 2025 2024 2023 2022 2021 2020"
GROUPS["group_mid"]="2019 2018 2017 2016 2015 2014 2013 2012"
GROUPS["group_old1"]="2011 2010 2009 2008 2007 2006 2005 2004"
GROUPS["group_old2"]="2003 2002 2001 2000 1999 1998 1997 1996 1995 1994 1993 1992 1991 1990 1989 1988 1987 1986 1985 1984 1983 1982 1981 1980 1979 1978 1977 1976 1975 1974 1973 1972 1971 1970 1969 1968 1967 1966 1965 1964 1963 1962 1961 1960 1959 1958 1957 1956 1955 1954 1953 1952 1951 1950"

for sess in group_recent group_mid group_old1 group_old2; do
    YEARS="${GROUPS[$sess]}"
    LOG="$LOG_DIR/${sess}_master.log"
    CMD="bash $SCRIPT $sess $YEARS 2>&1 | tee $LOG; echo 'SESSION $sess DONE'; bash"

    tmux new-session -d -s "$sess" -x 220 -y 50
    tmux send-keys -t "$sess" "cd /home/azureuser/masdb && source .env && export \$(grep -v '^#' .env | xargs)" Enter
    tmux send-keys -t "$sess" "$CMD" Enter

    echo "Launched tmux session: $sess  years: $YEARS"
done

echo ""
echo "=== All sessions launched ==="
tmux ls
echo ""
echo "Attach with:  tmux attach -t group_recent"
echo "Logs at:      $LOG_DIR/"

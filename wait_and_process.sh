#!/bin/bash
# Launch both scrapers simultaneously, and start each processor with --stream
# immediately so it overlaps with scraping from the very first act.
# 8 GPUs, 8 states. Two scrapers run in parallel (4 states each, sequential within each).

cd /home/azureuser/masdb
mkdir -p logs

SCRAPE_LOG=logs/scrape_state_acts.log
SCRAPE2_LOG=logs/scrape_state_acts_2.log

BATCH1=(karnataka delhi maharashtra haryana)
BATCH2=(gujarat rajasthan andhra_pradesh tamil_nadu)

GPU_MAP=(
    [karnataka]=0
    [delhi]=1
    [maharashtra]=2
    [haryana]=3
    [gujarat]=4
    [rajasthan]=5
    [andhra_pradesh]=6
    [tamil_nadu]=7
)

launch_stream_processor() {
    local state=$1
    local gpu=${GPU_MAP[$state]}
    local logfile="logs/proc_${state}.log"
    echo "$(date) [$state] Launching streaming processor on GPU ${gpu}"
    screen -dmS "proc_${state}" bash -c \
        "CUDA_VISIBLE_DEVICES=${gpu} .venv/bin/python pipeline/process_state_acts.py \
         --states ${state} --workers 4 --stream --stream-idle-secs 300 \
         > ${logfile} 2>&1"
    echo "$(date) [$state] proc_${state} screen started → ${logfile}"
}

# ── Start both scrapers simultaneously ──────────────────────────────────────
STATES1=$(IFS=,; echo "${BATCH1[*]}")
STATES2=$(IFS=,; echo "${BATCH2[*]}")

echo "$(date) Starting scraper 1: ${STATES1}"
screen -dmS scrape_states bash -c \
    ".venv/bin/python pipeline/scrape_state_acts.py --states ${STATES1} \
     > ${SCRAPE_LOG} 2>&1"

echo "$(date) Starting scraper 2: ${STATES2}"
screen -dmS scrape_states_2 bash -c \
    ".venv/bin/python pipeline/scrape_state_acts.py --states ${STATES2} \
     > ${SCRAPE2_LOG} 2>&1"

echo "$(date) Both scrapers running. Starting all 8 stream processors..."

# ── Launch all 8 processors immediately in streaming mode ───────────────────
# Each will poll its metadata.jsonl and start processing the moment the
# scraper writes the first act. --stream-idle-secs=300 means it exits
# after 5 minutes of no new acts (i.e. scraping for that state is done).
for state in "${BATCH1[@]}" "${BATCH2[@]}"; do
    launch_stream_processor "$state"
done

echo "$(date) All 8 processors launched in stream mode."
echo "Monitor:"
echo "  screen -ls                    # list all screens"
echo "  tail -f logs/proc_karnataka.log"
echo "  tail -f logs/scrape_state_acts.log"

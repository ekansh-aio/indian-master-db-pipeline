#!/bin/bash
# Round 2: scrape + process the 28 remaining states (all except the initial 8).
# 4 scrapers (7 states each, parallel), GPU queue scheduler handles 8 GPUs
# with 3-4 states each via watch_remaining.py.

cd /home/azureuser/masdb
mkdir -p logs

# ── 4 scraper batches (7 states each) ────────────────────────────────────────
BATCH1=(andaman_nicobar arunachal_pradesh assam bihar chandigarh chhattisgarh dadra_nagar_haveli_daman_diu)
BATCH2=(goa himachal_pradesh jammu_kashmir jharkhand kerala ladakh lakshadweep)
BATCH3=(madhya_pradesh manipur meghalaya mizoram nagaland odisha puducherry)
BATCH4=(punjab sikkim telangana tripura uttarakhand uttar_pradesh west_bengal)

echo "$(date) Starting 4 scrapers..."

STATES1=$(IFS=,; echo "${BATCH1[*]}")
screen -dmS scrape_r2_1 bash -c \
    ".venv/bin/python pipeline/scrape_state_acts.py --states ${STATES1} \
     > logs/scrape_r2_1.log 2>&1"
echo "  scrape_r2_1: ${STATES1}"

STATES2=$(IFS=,; echo "${BATCH2[*]}")
screen -dmS scrape_r2_2 bash -c \
    ".venv/bin/python pipeline/scrape_state_acts.py --states ${STATES2} \
     > logs/scrape_r2_2.log 2>&1"
echo "  scrape_r2_2: ${STATES2}"

STATES3=$(IFS=,; echo "${BATCH3[*]}")
screen -dmS scrape_r2_3 bash -c \
    ".venv/bin/python pipeline/scrape_state_acts.py --states ${STATES3} \
     > logs/scrape_r2_3.log 2>&1"
echo "  scrape_r2_3: ${STATES3}"

STATES4=$(IFS=,; echo "${BATCH4[*]}")
screen -dmS scrape_r2_4 bash -c \
    ".venv/bin/python pipeline/scrape_state_acts.py --states ${STATES4} \
     > logs/scrape_r2_4.log 2>&1"
echo "  scrape_r2_4: ${STATES4}"

echo "$(date) All 4 scrapers launched."

# ── GPU queue watcher (launches 8 processors, cycles through 28 states) ───────
echo "$(date) Starting GPU queue watcher (watch_remaining.py)..."
screen -dmS watcher_r2 bash -c \
    ".venv/bin/python watch_remaining.py \
     > logs/watcher_r2.log 2>&1"
echo "$(date) watcher_r2 screen started → logs/watcher_r2.log"

echo ""
echo "Monitor:"
echo "  screen -ls                         # list all screens"
echo "  tail -f logs/watcher_r2.log        # GPU queue progress"
echo "  tail -f logs/scrape_r2_1.log       # scraper 1"
echo "  tail -f logs/proc_uttar_pradesh.log"

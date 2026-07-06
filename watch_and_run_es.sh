#!/usr/bin/env bash
# watch_and_run_es.sh
#
# Watches hc_YEAR tmux sessions; as each finishes, enqueues
# adls_to_es_pipeline.py for that year. Runs ES jobs sequentially
# (one at a time — all 8 GPUs per job).
#
# Years watched: 2015, 2018, 2025  (2024 enqueued immediately — already done)
# Run in its own tmux session:
#   tmux new-session -d -s es_coord 'bash /home/azureuser/masdb/watch_and_run_es.sh 2>&1 | tee /home/azureuser/masdb/logs/es_coord.log'

set -uo pipefail

VENV="/home/azureuser/masdb/.venv/bin/python"
MASDB="/home/azureuser/masdb"
LOG_DIR="$MASDB/logs"
QUEUE_FILE="$LOG_DIR/es_queue.txt"
STATUS_FILE="$LOG_DIR/es_coord_status.txt"

mkdir -p "$LOG_DIR"
: > "$QUEUE_FILE"
: > "$STATUS_FILE"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$STATUS_FILE"; }

# ── Consumer (background) ────────────────────────────────────────────────────
# Reads queue file, runs one ES pipeline job at a time.
consumer() {
    log "Consumer started (PID $$)"
    while true; do
        if [ -s "$QUEUE_FILE" ]; then
            # Pop first line
            YEAR=$(head -1 "$QUEUE_FILE")
            tail -n +2 "$QUEUE_FILE" > "${QUEUE_FILE}.tmp" && mv "${QUEUE_FILE}.tmp" "$QUEUE_FILE"

            log ">>> ES pipeline START  year=$YEAR"
            ES_LOG="$LOG_DIR/es_hc_${YEAR}.log"
            cd "$MASDB"
            "$VENV" pipeline/adls_to_es_pipeline.py --doc-type hc --years "$YEAR" \
                2>&1 | tee "$ES_LOG"
            RC=${PIPESTATUS[0]}
            if [ $RC -eq 0 ]; then
                log "<<< ES pipeline DONE   year=$YEAR  (exit 0)"
            else
                log "<<< ES pipeline FAILED year=$YEAR  (exit $RC) — check $ES_LOG"
            fi
        else
            sleep 15
        fi
    done
}

consumer &
CONSUMER_PID=$!
log "Coordinator started  consumer_pid=$CONSUMER_PID"

# ── Enqueue 2024 immediately ─────────────────────────────────────────────────
echo "2024" >> "$QUEUE_FILE"
log "Enqueued year 2024 (hc_upload already complete)"

# ── Watcher ──────────────────────────────────────────────────────────────────
declare -A SESSIONS=([2015]="hc_2015" [2018]="hc_2018" [2025]="hc_2025")
declare -A ENQUEUED=()

log "Watching tmux sessions: hc_2015  hc_2018  hc_2025"

while [ ${#ENQUEUED[@]} -lt ${#SESSIONS[@]} ]; do
    for YEAR in "${!SESSIONS[@]}"; do
        [ -n "${ENQUEUED[$YEAR]:-}" ] && continue
        SESSION="${SESSIONS[$YEAR]}"
        if ! tmux has-session -t "$SESSION" 2>/dev/null; then
            ENQUEUED[$YEAR]=1
            log "Session $SESSION finished — enqueuing year $YEAR"
            echo "$YEAR" >> "$QUEUE_FILE"
        fi
    done
    sleep 30
done

log "All hc_upload sessions finished. Waiting for ES queue to drain ..."

# Wait for queue file to empty and consumer to finish last job
while [ -s "$QUEUE_FILE" ]; do
    sleep 30
done

# Give consumer time to finish running job (queue empty but job still running)
sleep 60
while true; do
    LAST=$(tail -1 "$STATUS_FILE" 2>/dev/null || echo "")
    if echo "$LAST" | grep -qE "DONE|FAILED"; then
        break
    fi
    sleep 30
done

log "All ES pipeline jobs complete."
log "Per-year logs: $LOG_DIR/es_hc_{year}.log"
log "Coordinator exiting."
kill "$CONSUMER_PID" 2>/dev/null || true

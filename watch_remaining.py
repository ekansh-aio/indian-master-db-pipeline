#!/usr/bin/env python3
"""
GPU queue scheduler for the 28 remaining state acts.
Each GPU is assigned a queue of states; when a processor finishes,
the next state in that GPU's queue is launched automatically.
"""
import subprocess
import sys
import time
from pathlib import Path

BASE_DIR = Path("/home/azureuser/masdb")
LOG_DIR  = BASE_DIR / "logs"
VENV     = BASE_DIR / ".venv/bin/python"

# Round-robin assignment: states[i] -> GPU (i % 8)
# Ordered alphabetically across 28 remaining states
GPU_QUEUES = {
    0: ["andaman_nicobar",           "himachal_pradesh",  "meghalaya",  "tripura"],
    1: ["arunachal_pradesh",         "jammu_kashmir",     "mizoram",    "uttarakhand"],
    2: ["assam",                     "jharkhand",         "nagaland",   "uttar_pradesh"],
    3: ["bihar",                     "kerala",            "odisha",     "west_bengal"],
    4: ["chandigarh",                "ladakh",            "puducherry"],
    5: ["chhattisgarh",              "lakshadweep",       "punjab"],
    6: ["dadra_nagar_haveli_daman_diu", "madhya_pradesh", "sikkim"],
    7: ["goa",                       "manipur",           "telangana"],
}


def ts():
    return time.strftime("%H:%M:%S")


def log(msg):
    print(f"{ts()} [watcher] {msg}", flush=True)


def screen_running(name):
    r = subprocess.run(["screen", "-ls"], capture_output=True, text=True)
    return f".{name}" in r.stdout


def launch_proc(state, gpu):
    log_file = LOG_DIR / f"proc_{state}.log"
    screen_name = f"proc_{state}"
    cmd = (
        f"CUDA_VISIBLE_DEVICES={gpu} {VENV} "
        f"pipeline/process_state_acts.py "
        f"--states {state} --workers 4 --stream --stream-idle-secs 900 "
        f"> {log_file} 2>&1"
    )
    subprocess.run(
        ["screen", "-dmS", screen_name, "bash", "-c", cmd],
        cwd=str(BASE_DIR),
    )
    log(f"Launched proc_{state} on GPU {gpu}")


def main():
    LOG_DIR.mkdir(exist_ok=True)

    gpu_current = {}
    gpu_queues  = {g: list(q) for g, q in GPU_QUEUES.items()}

    # Launch first state for each GPU immediately (stream mode: will wait for metadata)
    for gpu in range(8):
        queue = gpu_queues[gpu]
        if queue:
            state = queue.pop(0)
            launch_proc(state, gpu)
            gpu_current[gpu] = f"proc_{state}"
        else:
            gpu_current[gpu] = None

    total_pending = sum(len(q) for q in gpu_queues.values())
    log(f"All 8 initial processors launched. {total_pending} states queued behind them.")

    while True:
        time.sleep(30)
        any_active = False

        for gpu in range(8):
            current = gpu_current[gpu]
            if current is None:
                continue
            if not screen_running(current):
                log(f"{current} finished on GPU {gpu}")
                if gpu_queues[gpu]:
                    next_state = gpu_queues[gpu].pop(0)
                    launch_proc(next_state, gpu)
                    gpu_current[gpu] = f"proc_{next_state}"
                    any_active = True
                else:
                    gpu_current[gpu] = None
                    log(f"GPU {gpu} queue exhausted")
            else:
                any_active = True

        if not any_active:
            log("All 28 states processed. Exiting.")
            sys.exit(0)


if __name__ == "__main__":
    main()

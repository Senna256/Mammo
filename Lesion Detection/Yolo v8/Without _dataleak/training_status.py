#!/usr/bin/env python3
"""Status of the six Without _dataleak trainings: state, epoch, time per epoch and ETA.

Usage: python training_status.py
States: queued (tmux session waiting, no log yet), running, finished, stopped
(log exists but its tmux session is gone without "TRAINING FINISHED").
ETAs of queued runs chain after the run they wait for on the same GPU.
"""
import re
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
EPOCHS = 100

# (tmux session, script, estimated min/epoch before the run has finished an epoch) in GPU queue order.
QUEUES = {
    "GPU 0 (A40)": [
        ("vit_full_gpu0", "vit/Train/full_dataset/train_full_vit.py", 18.0),
        ("swin_full_gpu0", "swin/Train/full_dataset/train_full_swin.py", 12.0),
    ],
    "GPU 1 (A30)": [
        ("vit_4k_gpu1", "vit/Train/4k images/train_4k_vit.py", 6.5),
        ("swin_4k_gpu1", "swin/Train/4k images/train_4k_swin.py", 4.0),
        ("swin_positives_gpu1", "swin/Train/positives/train_swin_positives.py", 2.5),
        ("vit_positives_gpu1", "vit/Train/positives/train_vit_positives.py", 4.5),
    ],
}


def tmux_sessions():
    result = subprocess.run(["tmux", "ls", "-F", "#{session_name}"], capture_output=True, text=True)
    return set(result.stdout.split())


def log_status(log_path, started_after):
    """(finished?, epochs done, median min/epoch) of a log written by the current run, else None."""
    if not log_path.is_file() or datetime.fromtimestamp(log_path.stat().st_ctime) < started_after:
        return None
    text = log_path.read_text(errors="ignore")
    times = [float(t) for t in re.findall(r"^Time: +([0-9.]+) min", text, flags=re.M)]
    return "TRAINING FINISHED" in text, len(times), float(np.median(times[-10:])) if times else None


def main():
    sessions = tmux_sessions()
    started_after = datetime(2026, 10, 20, 11, 0)  # launch of the homogenised runs
    now = datetime.now()
    print(f"{now:%a %d %b %H:%M}\n")
    for gpu, queue in QUEUES.items():
        print(gpu)
        free_at = now
        for session, script, default_minutes in queue:
            script_path = ROOT / script
            status = log_status(script_path.with_suffix(".log"), started_after)
            finished, epochs_done, minutes = status if status else (False, 0, None)
            minutes = minutes or default_minutes
            if finished:
                state, eta = "finished", None
            elif status and session in sessions:
                state = "running"
                free_at = now + timedelta(minutes=minutes * (EPOCHS - epochs_done))
                eta = free_at
            elif session in sessions:
                state = "queued"
                free_at = free_at + timedelta(minutes=minutes * EPOCHS)
                eta = free_at
            else:
                state, eta = ("stopped (check log!)" if status else "not launched"), None
            eta_text = f"ETA {eta:%a %d %b %H:%M}" if eta else ""
            print(f"  {script_path.parent.parent.parent.name:4s} {script_path.parent.name:13s} {state:20s} "
                  f"epoch {epochs_done:3d}/{EPOCHS}  {minutes:5.1f} min/epoch  {eta_text}")
        print()


if __name__ == "__main__":
    main()

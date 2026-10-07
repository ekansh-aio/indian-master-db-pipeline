"""
State Acts Scraper — downloads PDFs + metadata from indiacode.nic.in per state.

Callable standalone or via run.py:
    python pipelines/state_acts/scrape.py --states karnataka,delhi
    python pipelines/state_acts/scrape.py --all-states
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from pipeline.scrape_state_acts import main

if __name__ == "__main__":
    main()

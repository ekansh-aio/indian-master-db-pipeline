"""
Central Acts Scraper — downloads PDFs + metadata from indiacode.nic.in.

Callable standalone or via run.py:
    python pipelines/central_acts/scrape.py [--output-dir central_acts_pdfs] [--metadata-only]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# Re-export the original implementation
from pipeline.scrape_central_acts import main

if __name__ == "__main__":
    main()

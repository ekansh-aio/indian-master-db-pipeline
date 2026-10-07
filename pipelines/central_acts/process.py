"""
Central Acts Processor — PDF → text → semantic chunks → embeddings → ADLS.

Callable standalone or via run.py:
    python pipelines/central_acts/process.py [--pdf-dir central_acts_pdfs] [--no-resume]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from pipeline.process_central_acts import main

if __name__ == "__main__":
    main()

"""Portable project paths.

Override ``LIFT_OUTPUT_DIR`` when results should be written outside the repo.
"""

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
result_dir = os.environ.get("LIFT_OUTPUT_DIR", str(PROJECT_ROOT / "results"))

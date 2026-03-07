from __future__ import annotations

import runpy
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
STREAMLIT_DIR = ROOT_DIR / "streamlit"
STREAMLIT_APP = STREAMLIT_DIR / "app.py"

if str(STREAMLIT_DIR) not in sys.path:
    sys.path.insert(0, str(STREAMLIT_DIR))

runpy.run_path(str(STREAMLIT_APP), run_name="__main__")

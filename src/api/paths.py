"""
Centralised path constants for the DE BRUITS application.

All other modules import from here so path resolution is defined in one place.
"""

import os
from pathlib import Path

_API_DIR     = Path(__file__).resolve().parent   # src/api/
SRC_DIR      = _API_DIR.parent                   # src/
PROJECT_ROOT = SRC_DIR.parent                    # repo root (or /app in Docker)

STATIC_DIR  = SRC_DIR / "static"
FONTS_DIR   = SRC_DIR / "fonts"
THUMBS_PATH = SRC_DIR / "thumbnails.json"

STAMP_EXTRA_BOTTOM = 0.08  # fraction of page height reserved below stamp

DATA_ROOT = Path(os.environ.get("DATA_ROOT", str(PROJECT_ROOT / "data")))

# config.json lives in DATA_ROOT; _CFG_SEED is the default shipped inside the image
_CFG_SEED = SRC_DIR / "config.json"
CFG_PATH  = DATA_ROOT / "config.json"
ICONS_DIR = DATA_ROOT / "Icons"

TMPL_DIR  = DATA_ROOT / "Templates"
FINAL_DIR = DATA_ROOT / "Series"
COVER_DIR = DATA_ROOT / "Cover"
ZINES_DIR = DATA_ROOT / "Zines"

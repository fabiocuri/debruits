"""
Centralised path constants for the DE BRUITS application.

All other modules import from here so path resolution is defined in one place.
"""

import json
import os
from pathlib import Path

_API_DIR     = Path(__file__).resolve().parent   # src/api/
SRC_DIR      = _API_DIR.parent                   # src/
PROJECT_ROOT = SRC_DIR.parent                    # repo root (or /app in Docker)

CFG_PATH    = SRC_DIR / "config.json"
STATIC_DIR  = SRC_DIR / "static"
FONTS_DIR   = SRC_DIR / "fonts"
ICONS_DIR   = PROJECT_ROOT / "icons"
THUMBS_PATH = SRC_DIR / "thumbnails.json"

STAMP_EXTRA_BOTTOM = 0.08  # fraction of page height reserved below stamp


def _load_data_root() -> Path:
    env = os.environ.get("DATA_ROOT")
    if env:
        return Path(env)
    try:
        with open(CFG_PATH) as f:
            cfg = json.load(f)
        if cfg.get("data_root"):
            return Path(cfg["data_root"])
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    return PROJECT_ROOT / "data"


DATA_ROOT = _load_data_root()
TMPL_DIR  = DATA_ROOT / "Templates"
FINAL_DIR = DATA_ROOT / "Series"
COVER_DIR = DATA_ROOT / "Cover"
ZINES_DIR = DATA_ROOT / "Zines"

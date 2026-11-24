"""Shared utilities used across route modules."""

import json
import re
from datetime import datetime
from pathlib import Path

import img2pdf
from flask import send_file
from PIL import Image

from .paths import _CFG_SEED, CFG_PATH, DATA_ROOT, SRC_DIR, THUMBS_PATH

_LANG_TO_DIR = {"pt": "PORTUGUES", "en": "ENGLISH", "es": "ESPANOL", "fr": "FRANCAIS"}
_DIR_TO_LANG = {v: k for k, v in _LANG_TO_DIR.items()}

_GEO_PREP = re.compile(
    r'\b(DE|DAS|DOS|FROM|VON|VAN|DI|DES|DEL|DEN|DEGLI|DU)\b',
    re.IGNORECASE,
)


def lang_to_dir(lang: str) -> str:
    return _LANG_TO_DIR.get(lang, lang.upper())


def dir_to_lang(d: str) -> str:
    return _DIR_TO_LANG.get(d.upper(), "pt")


def auto_split_title(title: str) -> str:
    """Insert newline before the last geographic preposition when >1 word precedes it."""
    title = " ".join(title.splitlines()).strip()
    matches = list(_GEO_PREP.finditer(title))
    if matches:
        m = matches[-1]
        before = title[: m.start()].strip()
        if len(before.split()) > 1:
            return before + "\n" + title[m.start():].strip()
    return title


# ── Config ─────────────────────────────────────────────────────────────────────

def load_cfg() -> dict:
    if not CFG_PATH.exists() and _CFG_SEED.exists():
        import shutil
        CFG_PATH.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(_CFG_SEED, CFG_PATH)
    with open(CFG_PATH) as f:
        return json.load(f)


def save_cfg(cfg: dict) -> None:
    with open(CFG_PATH, "w") as f:
        json.dump(cfg, f, indent=2)


def normalize_titles(cfg: dict) -> dict:
    """Ensure series_titles[series][lang] is always a list of strings."""
    for langs in cfg.get("series_titles", {}).values():
        for lang, v in list(langs.items()):
            if isinstance(v, str):
                langs[lang] = [v] if v else []
            elif not isinstance(v, list):
                langs[lang] = []
    return cfg


def flatten_titles(raw_titles: dict) -> dict:
    """Collapse series_titles lists to a single string per lang."""
    return {
        s: {l: (v[0] if isinstance(v, list) else v) for l, v in langs.items()}
        for s, langs in raw_titles.items()
    }


# ── Thumbnails ─────────────────────────────────────────────────────────────────

def load_thumbnails() -> dict:
    try:
        with open(THUMBS_PATH) as f:
            data = json.load(f)
        migrated = {
            k: v.replace("Final/", "Series/", 1) if v.startswith("Final/") else v
            for k, v in data.items()
        }
        if migrated != data:
            with open(THUMBS_PATH, "w") as f:
                json.dump(migrated, f, indent=2)
        return migrated
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


# ── File metadata ──────────────────────────────────────────────────────────────

def img_meta(path: Path) -> dict:
    stat = path.stat()
    try:
        with Image.open(path) as img:
            w, h = img.size
    except Exception:
        w = h = 0
    return {
        "size_bytes": stat.st_size,
        "modified":   datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
        "width": w,
        "height": h,
    }


def parse_template(path: Path) -> dict | None:
    if path.suffix != ".pdf":
        return None
    sidecar = path.with_suffix(".json")
    if not sidecar.exists():
        return None
    try:
        meta = json.loads(sidecar.read_text())
    except Exception:
        return None
    series = meta.get("series", "")
    if not series:
        return None
    thumb = path.with_name(path.stem + "_thumb.png")
    return {
        "format": meta.get("format", "postcard"),
        "series": series.upper(),
        "lang":   meta.get("lang", "pt"),
        "icon":   meta.get("icon"),
        "bw":     bool(meta.get("bw", False)),
        "thumb":  str(thumb.relative_to(DATA_ROOT)) if thumb.exists() else None,
        "page":   None,
    }


# ── File serving ───────────────────────────────────────────────────────────────

def safe_send(path: Path):
    resolved = path.resolve()
    allowed = {str(SRC_DIR.resolve()), str(DATA_ROOT.resolve())}
    if not any(str(resolved).startswith(a) for a in allowed):
        return "Forbidden", 403
    if not resolved.exists():
        return "Not found", 404
    return send_file(resolved)


def png_to_pdf_bytes(path: Path) -> bytes:
    return img2pdf.convert(str(path))


# ── Image helpers ──────────────────────────────────────────────────────────────

def fit_and_rotate(img: Image.Image, w: int, h: int, rot: float) -> Image.Image:
    """Fit img into w x h (object-fit: contain), then rotate in-place."""
    ratio = min(w / img.width, h / img.height)
    fw = max(1, round(img.width  * ratio))
    fh = max(1, round(img.height * ratio))
    resized = img.resize((fw, fh), Image.LANCZOS)
    cell = Image.new("RGB", (w, h), (255, 255, 255))
    cell.paste(resized, ((w - fw) // 2, (h - fh) // 2))
    if rot:
        cell = cell.rotate(-rot, expand=False, resample=Image.BICUBIC,
                           fillcolor=(255, 255, 255))
    return cell


def fit_and_rotate_rgba(img_rgba: Image.Image, w: int, h: int, rot: float) -> Image.Image:
    """Like fit_and_rotate but preserves alpha."""
    ratio = min(w / img_rgba.width, h / img_rgba.height)
    fw = max(1, round(img_rgba.width  * ratio))
    fh = max(1, round(img_rgba.height * ratio))
    resized = img_rgba.resize((fw, fh), Image.LANCZOS)
    r, g, b_ch, a = resized.split()
    a = a.point(lambda v: 255 if v > 127 else 0)
    resized = Image.merge("RGBA", (r, g, b_ch, a))
    cell = Image.new("RGBA", (w, h), (255, 255, 255, 0))
    cell.paste(resized, ((w - fw) // 2, (h - fh) // 2), mask=a)
    if rot:
        cell = cell.rotate(-rot, expand=False, resample=Image.BICUBIC)
    return cell

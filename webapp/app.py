#!/usr/bin/env python3
"""
DE BRUITS — project dashboard.

Usage:
    pip install flask pillow
    python3 app.py
    → open http://localhost:5000

The interests-info-dashboard is launched automatically as a subprocess on port 5001.
"""

import atexit
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

try:
    from flask import Flask, jsonify, request, send_file
except ImportError:
    sys.exit("Flask not found. Run: pip install flask")

from PIL import Image

ROOT       = Path(__file__).resolve().parent
CFG_PATH   = ROOT / "config.json"
STATIC_DIR = ROOT / "static"

# ── Load data root from config at startup ──────────────────────────────────────
def _boot_cfg() -> dict:
    try:
        with open(CFG_PATH) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

_cfg       = _boot_cfg()
DATA_ROOT  = Path(_cfg.get("data_root", ROOT.parent.parent))
TMPL_DIR   = DATA_ROOT / "Templates"
FINAL_DIR  = DATA_ROOT / "Final"

app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path="/static")


# ── Interests dashboard subprocess ─────────────────────────────────────────────

_interests_proc = None

def _start_interests_dashboard() -> None:
    global _interests_proc
    dashboard_path = ROOT / "interests"
    if not dashboard_path.exists():
        print(f"  Info tab  : interests/ not found — skipping")
        return
    info_url = _cfg.get("info_dashboard_url", "http://localhost:5001")
    port     = info_url.rsplit(":", 1)[-1] if ":" in info_url else "5001"
    _interests_proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.main:app", "--port", port],
        cwd=str(dashboard_path),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    print(f"  Info tab  : interests dashboard → {info_url} (pid {_interests_proc.pid})")

def _stop_interests_dashboard() -> None:
    if _interests_proc and _interests_proc.poll() is None:
        _interests_proc.terminate()

atexit.register(_stop_interests_dashboard)


# ── Config ─────────────────────────────────────────────────────────────────────

def load_cfg() -> dict:
    with open(CFG_PATH) as f:
        return json.load(f)

def save_cfg(cfg: dict) -> None:
    with open(CFG_PATH, "w") as f:
        json.dump(cfg, f, indent=2)

@app.route("/api/config", methods=["GET", "PUT"])
def api_config():
    if request.method == "PUT":
        save_cfg(request.json)
        return jsonify({"ok": True})
    return jsonify(load_cfg())


# ── File serving ───────────────────────────────────────────────────────────────

def _safe_send(path: Path):
    resolved = path.resolve()
    allowed  = {str(ROOT.resolve()), str(DATA_ROOT.resolve())}
    if not any(str(resolved).startswith(a) for a in allowed):
        return "Forbidden", 403
    if not resolved.exists():
        return "Not found", 404
    return send_file(resolved)

@app.route("/img/<path:fp>")
def serve_img(fp):
    return _safe_send(DATA_ROOT / fp)

@app.route("/font/<name>")
def serve_font(name):
    return _safe_send(ROOT / name)


# ── Template listing ───────────────────────────────────────────────────────────

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
        "width": w, "height": h,
    }


def parse_template(path: Path) -> dict | None:
    try:
        parts = path.relative_to(TMPL_DIR).parts
    except ValueError:
        return None
    if len(parts) < 2:
        return None

    fmt  = parts[0].lower()
    stem = path.stem

    if fmt == "fanzine":
        if len(parts) < 3:
            return None
        series = parts[1].upper()
        page   = int(stem) if stem.isdigit() else None
        return {"format": "fanzine", "series": series, "side": "page", "page": page, "lang": None}

    lang_dir = parts[1].upper() if len(parts) > 2 else "PORTUGUES"
    lang     = "pt" if lang_dir == "PORTUGUES" else "en"

    if stem.endswith("_FRONT"):
        series, side = stem[:-6], "front"
    elif "_BACK" in stem:
        series = stem.rsplit("_BACK", 1)[0]
        side   = "back"
    else:
        series = re.split(r"[_.]", stem)[0]
        side   = "front"

    return {"format": fmt, "series": series.upper(), "lang": lang, "side": side, "page": None}


@app.route("/api/templates")
def api_templates():
    out = []
    for path in sorted(TMPL_DIR.rglob("*.png")):
        meta = parse_template(path)
        if meta is None:
            continue
        meta["path"] = str(path.relative_to(DATA_ROOT))
        meta.update(img_meta(path))
        out.append(meta)
    return jsonify(out)


# ── Final image listing ────────────────────────────────────────────────────────

@app.route("/api/images")
def api_images():
    out = []
    for path in sorted(FINAL_DIR.glob("*.png")):
        m   = re.match(r"^([A-Z]+)_(\d+)", path.stem)
        rec = {
            "path":     str(path.relative_to(DATA_ROOT)),
            "filename": path.name,
            "series":   m.group(1) if m else path.stem.split("_")[0].upper(),
            "index":    int(m.group(2)) if m else 0,
        }
        rec.update(img_meta(path))
        out.append(rec)
    return jsonify(out)


# ── Generation ─────────────────────────────────────────────────────────────────

@app.route("/api/generate", methods=["POST"])
def api_generate():
    data   = request.json or {}
    series = data.get("series", "").strip().upper()
    fmt    = data.get("format", "postcard")
    lang   = data.get("lang", "pt")
    side   = data.get("side", "both")

    if not series:
        return jsonify({"ok": False, "error": "series is required"}), 400

    script = ROOT / "make_template.py"
    result = subprocess.run(
        [sys.executable, str(script), series, fmt, lang, "--side", side],
        capture_output=True, text=True, cwd=ROOT
    )
    return jsonify({"ok": result.returncode == 0,
                    "stdout": result.stdout, "stderr": result.stderr})


# ── Entry point ────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_file(STATIC_DIR / "index.html")

if __name__ == "__main__":
    print(f"\nDE BRUITS dashboard → http://localhost:5000")
    print(f"  Data root : {DATA_ROOT}")
    _start_interests_dashboard()
    print()
    app.run(debug=False, port=5000)

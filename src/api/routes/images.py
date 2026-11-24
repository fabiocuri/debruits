"""Series image listing, thumbnails, and cover photo routes."""

import json
import re

from flask import Blueprint, jsonify, request

from ..paths import COVER_DIR, DATA_ROOT, FINAL_DIR, THUMBS_PATH
from ..utils import img_meta, load_thumbnails

bp = Blueprint("images", __name__)


@bp.route("/api/images")
def api_images():
    out = []
    for path in sorted(FINAL_DIR.rglob("*.png")):
        parent = path.parent
        series = (parent.name.upper() if parent != FINAL_DIR
                  else path.stem.split("_")[0].upper())
        m = re.match(r"^[A-Z]+_(\d+)", path.stem)
        rec = {
            "path":     str(path.relative_to(DATA_ROOT)),
            "filename": path.name,
            "series":   series,
            "index":    int(m.group(1)) if m else 0,
        }
        rec.update(img_meta(path))
        out.append(rec)
    return jsonify(out)


@bp.route("/api/covers/<series>")
def api_covers(series):
    folder = COVER_DIR / series.upper()
    if not folder.exists():
        return jsonify([])
    exts = {".png", ".jpg", ".jpeg", ".webp"}
    return jsonify([
        {"path": str(p.relative_to(DATA_ROOT))}
        for p in sorted(folder.iterdir())
        if p.suffix.lower() in exts
    ])


@bp.route("/api/thumbnails", methods=["GET", "PUT"])
def api_thumbnails():
    if request.method == "PUT":
        data   = request.json or {}
        series = data.get("series", "").strip().upper()
        path   = data.get("path", "").strip()
        if series and path:
            thumbs = load_thumbnails()
            thumbs[series] = path
            with open(THUMBS_PATH, "w") as f:
                json.dump(thumbs, f, indent=2)
        return jsonify({"ok": True})
    return jsonify(load_thumbnails())

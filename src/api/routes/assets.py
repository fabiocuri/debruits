"""File-serving routes: images, fonts, icons, PDF downloads."""

import io
import zipfile
from pathlib import Path

from flask import Blueprint, jsonify, request, send_file

from ..paths import DATA_ROOT, FONTS_DIR, ICONS_DIR, TMPL_DIR
from ..utils import png_to_pdf_bytes, safe_send

bp = Blueprint("assets", __name__)


@bp.route("/img/<path:fp>")
def serve_img(fp):
    resp = safe_send(DATA_ROOT / fp)
    if isinstance(resp, tuple):
        return resp
    if fp.startswith("Zines/"):
        resp.headers["Cache-Control"] = "no-store"
        resp.headers.pop("ETag", None)
        resp.headers.pop("Last-Modified", None)
    return resp


@bp.route("/font/<name>")
def serve_font(name):
    return safe_send(FONTS_DIR / name)


@bp.route("/icon/<name>")
def serve_icon(name):
    return safe_send(ICONS_DIR / name)


@bp.route("/api/icons")
def api_icons():
    if not ICONS_DIR.exists():
        return jsonify([])
    return jsonify([
        {"name": p.name, "url": f"icon/{p.name}"}
        for p in sorted(ICONS_DIR.glob("*.png"))
    ])


@bp.route("/api/download/<path:fp>")
def api_download(fp):
    path = (DATA_ROOT / fp).resolve()
    if not str(path).startswith(str(DATA_ROOT.resolve())):
        return "Forbidden", 403
    if not path.exists():
        return "Not found", 404
    if path.suffix == ".pdf":
        return send_file(path, mimetype="application/pdf", as_attachment=True,
                         download_name=path.name)
    buf = io.BytesIO(png_to_pdf_bytes(path))
    return send_file(buf, mimetype="application/pdf", as_attachment=True,
                     download_name=path.stem + ".pdf")


@bp.route("/api/download-all")
def api_download_all():
    paths = sorted(TMPL_DIR.rglob("*.pdf"))
    if not paths:
        return jsonify({"error": "No saved prints found"}), 404
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in paths:
            try:
                zf.write(path, arcname=path.name)
            except Exception:
                pass
    buf.seek(0)
    return send_file(buf, mimetype="application/zip", as_attachment=True,
                     download_name="DE_BRUITS_HQ.zip")

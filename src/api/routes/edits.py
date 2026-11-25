"""Per-image brightness/contrast edit store."""

from flask import Blueprint, jsonify, request

from ..utils import load_edits, save_edits

bp = Blueprint("edits", __name__)


@bp.route("/api/edits", methods=["GET"])
def get_edits():
    return jsonify(load_edits())


@bp.route("/api/edits", methods=["PUT"])
def update_edit():
    data = request.json or {}
    path = data.get("path", "").strip()
    if not path:
        return jsonify({"ok": False, "error": "missing path"}), 400
    edits = load_edits()
    edits[path] = {
        "brightness": float(data.get("brightness", 100)),
        "contrast":   float(data.get("contrast",   100)),
    }
    save_edits(edits)
    return jsonify({"ok": True})


@bp.route("/api/edits", methods=["DELETE"])
def delete_edit():
    data = request.json or {}
    path = data.get("path", "").strip()
    if not path:
        return jsonify({"ok": False, "error": "missing path"}), 400
    edits = load_edits()
    edits.pop(path, None)
    save_edits(edits)
    return jsonify({"ok": True})

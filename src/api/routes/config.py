"""Application config read/write routes."""

from flask import Blueprint, jsonify, request

from .. import compositor as _mt
from ..utils import load_cfg, normalize_titles, save_cfg

bp = Blueprint("config", __name__)


@bp.route("/api/config", methods=["GET", "PUT"])
def api_config():
    if request.method == "PUT":
        save_cfg(request.json)
        return jsonify({"ok": True})
    cfg = normalize_titles(load_cfg())
    st = cfg.setdefault("series_titles", {})
    for series, langs in _mt.SERIES_TITLES.items():
        for lang, default_title in langs.items():
            if default_title:
                arr = st.setdefault(series, {}).setdefault(lang, [])
                if not arr:
                    arr.insert(0, default_title)
    return jsonify(cfg)

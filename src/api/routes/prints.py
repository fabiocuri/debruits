"""Print template routes: list, generate preview, compose + save, refresh, delete."""

import json
from datetime import datetime
from io import BytesIO
from pathlib import Path

import img2pdf
from flask import Blueprint, jsonify, request
from PIL import Image
from pypdf import PdfReader, PdfWriter

from .. import compositor as _mt
from ..paths import DATA_ROOT, FINAL_DIR, ICONS_DIR, STAMP_EXTRA_BOTTOM, TMPL_DIR
from ..utils import (
    flatten_titles,
    img_meta,
    lang_to_dir,
    load_cfg,
    normalize_titles,
    parse_template,
    save_cfg,
)

bp = Blueprint("prints", __name__)


# ── Private helpers ────────────────────────────────────────────────────────────

def _stamp_pos(W: int, H: int, icon_scale: float, has_icon: bool) -> dict:
    """Compute stamp bounding box in canvas pixels."""
    L      = _mt.LAYOUT
    s_size = round(round(L["stamp_size"] * W) * (icon_scale if has_icon else 1.0))
    s_x1   = W - round(L["stamp_right"]  * W)
    s_y1   = H - round(L["stamp_bottom"] * H) - round(STAMP_EXTRA_BOTTOM * H)
    return {"x": s_x1 - s_size, "y": s_y1 - s_size, "size": s_size}


def _build_placements(layout: list, bw: bool) -> list:
    """Convert layout items from the API request to compositor placement dicts."""
    return [
        {
            "path":       str(DATA_ROOT / item["url"]),
            "x":          int(item["x"]),
            "y":          int(item["y"]),
            "w":          int(item["w"]),
            "h":          int(item["h"]),
            "rot":        float(item.get("rot", 0)),
            "brightness": float(item.get("brightness", 100)),
            "contrast":   float(item.get("contrast",   100)),
            "bw":         bw,
        }
        for item in layout
    ]


def _images_to_pdf(images: list, size_px: tuple) -> bytes:
    """Render a list of PIL images to a multi-page PDF at the given pixel size."""
    W, H    = size_px
    page_fn = img2pdf.get_layout_fun(
        pagesize=(W / 300 * 72, H / 300 * 72), fit=img2pdf.FitMode.exact)
    writer  = PdfWriter()
    for img in images:
        buf = BytesIO()
        img.convert("RGB").save(buf, "PNG", compress_level=1)
        writer.add_page(
            PdfReader(BytesIO(img2pdf.convert(buf.getvalue(), layout_fun=page_fn))).pages[0])
    out = BytesIO()
    writer.write(out)
    return out.getvalue()


# ── Routes ─────────────────────────────────────────────────────────────────────

@bp.route("/api/templates", methods=["DELETE"])
def api_delete_templates():
    paths     = (request.json or {}).get("paths", [])
    deleted   = []
    tmpl_root = TMPL_DIR.resolve()
    for rel in paths:
        p = (DATA_ROOT / rel).resolve()
        if str(p).startswith(str(tmpl_root)) and p.exists() and p.suffix == ".pdf":
            p.unlink()
            deleted.append(rel)
            for extra in (p.with_suffix(".json"), p.with_name(p.stem + "_thumb.png")):
                if extra.exists():
                    extra.unlink()
    return jsonify({"ok": True, "deleted": deleted})


@bp.route("/api/templates")
def api_templates():
    out = []
    for path in sorted(TMPL_DIR.rglob("*.pdf")):
        meta = parse_template(path)
        if meta is None:
            continue
        meta["path"] = str(path.relative_to(DATA_ROOT))
        thumb_abs = DATA_ROOT / meta["thumb"] if meta.get("thumb") else None
        if thumb_abs and thumb_abs.exists():
            meta.update(img_meta(thumb_abs))
        else:
            stat = path.stat()
            meta.update({
                "size_bytes": stat.st_size,
                "modified":   datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
                "width": 0, "height": 0,
            })
        out.append(meta)
    return jsonify(out)


@bp.route("/api/layout-data/<path:fp>")
def api_layout_data(fp):
    path     = (DATA_ROOT / fp).with_suffix(".json").resolve()
    tmpl_res = TMPL_DIR.resolve()
    if not str(path).startswith(str(tmpl_res)):
        return "Forbidden", 403
    if not path.exists():
        return jsonify(None)
    meta      = json.loads(path.read_text())
    icon_name = meta.get("icon")
    size_px   = meta.get("size_px")
    if icon_name and size_px and len(size_px) == 2:
        meta["stamp"] = _stamp_pos(size_px[0], size_px[1],
                                    float(meta.get("iconScale", 1.0)), True)
    return jsonify(meta)


@bp.route("/api/generate", methods=["POST"])
def api_generate():
    data           = request.json or {}
    series         = data.get("series", "").strip().upper()
    fmt            = data.get("format", "postcard")
    count          = data.get("count")
    icon_name      = data.get("icon")
    icon_path      = str(ICONS_DIR / icon_name) if icon_name else None
    icon_scale     = float(data.get("iconScale", 1.0))
    selected_image = data.get("selectedImage")
    forced_path    = str(DATA_ROOT / selected_image) if selected_image else None

    if not series:
        return jsonify({"ok": False, "error": "series is required"}), 400

    blank_theme = (series == "__BLANK__")

    try:
        size_px = _mt.FORMATS.get(fmt, _mt.FORMATS["postcard"])["size_px"]
        W, H    = size_px

        if blank_theme:
            layout_out = []
        else:
            if count is None:
                count = len([p for p in FINAL_DIR.rglob("*.png")
                             if p.parent.name.upper() == series
                             or p.stem.split("_")[0].upper() == series])
            count = max(int(count), 1)

            _canvas, raw_layout = _mt.build_front_scattered(
                series, size_px, count,
                icon_path=icon_path, icon_scale=icon_scale, forced_path=forced_path,
            )
            layout_out = []
            for item in raw_layout:
                p = Path(item["path"])
                try:
                    url = str(p.relative_to(DATA_ROOT))
                except ValueError:
                    url = str(p)
                layout_out.append({"url": url, "x": item["x"], "y": item["y"],
                                   "w": item["w"], "h": item["h"]})

        return jsonify({
            "ok":          True,
            "paths":       [],
            "previewOnly": True,
            "size_px":     list(size_px),
            "layout":      layout_out,
            "stamp":       _stamp_pos(W, H, icon_scale, bool(icon_name)),
            "icon":        icon_name,
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@bp.route("/api/compose", methods=["POST"])
def api_compose():
    data         = request.json or {}
    layout       = data.get("layout", [])
    series       = data.get("series", "").strip().upper()
    fmt          = data.get("format", "postcard")
    lang         = data.get("lang", "pt")
    icon_name    = data.get("icon")
    icon_path    = str(ICONS_DIR / icon_name) if icon_name else None
    icon_scale   = float(data.get("iconScale", 1.0))
    bw           = bool(data.get("bw", False))
    custom_title = data.get("customTitle") or None
    blank_theme  = (series == "__BLANK__")

    if not series or (not layout and not blank_theme):
        return jsonify({"ok": False, "error": "layout and series required"}), 400

    try:
        size_px  = _mt.FORMATS.get(fmt, _mt.FORMATS["postcard"])["size_px"]
        W, H     = size_px
        out_dir  = TMPL_DIR / fmt.upper() / lang_to_dir(lang)
        out_dir.mkdir(parents=True, exist_ok=True)
        icon_stem = icon_name.rsplit(".", 1)[0] if icon_name else None
        base_name = (f"blank_{icon_stem}" if icon_stem else "blank") if blank_theme \
                    else (f"{series}_{icon_stem}" if icon_stem else series)

        cfg         = load_cfg()
        front_img   = _mt.compose_layout(_build_placements(layout, bw), size_px,
                                         icon_path=icon_path, icon_scale=icon_scale)
        back_img    = _mt.build_back("" if blank_theme else series, lang, size_px,
                                     title="" if blank_theme else custom_title,
                                     footer_text=cfg.get("back_footer", {}).get(lang))

        if custom_title and not blank_theme:
            cfg = normalize_titles(cfg)
            opts = cfg.setdefault("series_titles", {}).setdefault(series, {}).setdefault(lang, [])
            if custom_title not in opts:
                opts.append(custom_title)
            save_cfg(cfg)

        pdf_path   = out_dir / f"{base_name}.pdf"
        thumb_path = out_dir / f"{base_name}_thumb.png"
        pdf_path.write_bytes(_images_to_pdf([front_img, back_img], size_px))

        thumb_h = round(H * 500 / W)
        front_img.resize((500, thumb_h), Image.LANCZOS).save(
            str(thumb_path), "PNG", compress_level=6)

        stamp      = _stamp_pos(W, H, icon_scale, bool(icon_name))
        layout_out = [
            {"url": item["url"], "x": item["x"], "y": item["y"],
             "w": item["w"], "h": item["h"], "rot": item.get("rot", 0),
             "brightness": item.get("brightness", 100), "contrast": item.get("contrast", 100)}
            for item in layout
        ]
        (out_dir / f"{base_name}.json").write_text(json.dumps({
            "series": series, "format": fmt, "lang": lang, "icon": icon_name,
            "iconScale": icon_scale, "bw": bw, "layout": layout_out,
            "size_px": list(size_px), "stamp": stamp,
        }))

        return jsonify({
            "ok":    True,
            "paths": [str(pdf_path.relative_to(DATA_ROOT))],
            "thumb": str(thumb_path.relative_to(DATA_ROOT)),
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@bp.route("/api/refresh-prints", methods=["POST"])
def api_refresh_prints():
    cfg           = load_cfg()
    back_footers  = cfg.get("back_footer", {})
    series_titles = flatten_titles(cfg.get("series_titles", {}))

    count, errors = 0, []
    for json_path in sorted(TMPL_DIR.rglob("*.json")):
        try:
            with open(json_path) as f:
                sidecar = json.load(f)
        except Exception:
            continue
        series      = sidecar.get("series", "").strip().upper()
        lang        = sidecar.get("lang", "pt")
        fmt         = sidecar.get("format", "postcard")
        icon_name   = sidecar.get("icon")
        icon_scale  = float(sidecar.get("iconScale", 1.0))
        bw          = bool(sidecar.get("bw", False))
        layout      = sidecar.get("layout", [])
        size_px     = tuple(sidecar.get(
            "size_px", list(_mt.FORMATS.get(fmt, _mt.FORMATS["postcard"])["size_px"])))
        blank_theme = (series == "__BLANK__")
        if not series or (not layout and not blank_theme):
            continue
        try:
            icon_path = str(ICONS_DIR / icon_name) if icon_name else None
            front_img = _mt.compose_layout(_build_placements(layout, bw), size_px,
                                           icon_path=icon_path, icon_scale=icon_scale)
            back_img  = _mt.build_back(
                "" if blank_theme else series, lang, size_px,
                title="" if blank_theme else series_titles.get(series, {}).get(lang),
                footer_text=back_footers.get(lang),
            )
            W, H = size_px
            json_path.with_suffix(".pdf").write_bytes(
                _images_to_pdf([front_img, back_img], size_px))
            thumb_h = round(H * 500 / W)
            front_img.resize((500, thumb_h), Image.LANCZOS).save(
                str(json_path.with_name(json_path.stem + "_thumb.png")), "PNG", compress_level=6)
            count += 1
        except Exception as e:
            errors.append(f"{json_path.stem}: {e}")
    return jsonify({"ok": True, "count": count, "errors": errors})

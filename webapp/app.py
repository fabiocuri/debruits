#!/usr/bin/env python3
"""
DE BRUITS — project dashboard.

Usage:
    pip install flask pillow
    python3 app.py
    → open http://localhost:5000
"""

import io
import json
import re
import subprocess
import sys
import zipfile
import img2pdf
from datetime import datetime
from pathlib import Path

try:
    from flask import Flask, jsonify, request, send_file
except ImportError:
    sys.exit("Flask not found. Run: pip install flask")

from PIL import Image, ImageDraw
import make_template as _mt

_GEO_PREP = re.compile(
    r'\b(DE|DAS|DOS|FROM|VON|VAN|DI|DES|DEL|DEN|DEGLI|DU)\b',
    re.IGNORECASE,
)


def _auto_split_title(title: str, draw=None, font=None, max_w: int = 0) -> str:
    """Split title at last geographic preposition when >1 word precedes it."""
    title = ' '.join(title.splitlines()).strip()
    matches = list(_GEO_PREP.finditer(title))
    if matches:
        m = matches[-1]
        before = title[:m.start()].strip()
        if len(before.split()) > 1:
            return before + '\n' + title[m.start():].strip()
    return title


def save_png(img: Image.Image, path) -> None:
    img.convert("RGB").save(path, "PNG", dpi=(600, 600), compress_level=1)

_LANG_TO_DIR = {"pt": "PORTUGUES", "en": "ENGLISH", "es": "ESPANOL", "fr": "FRANCAIS"}
_DIR_TO_LANG = {v: k for k, v in _LANG_TO_DIR.items()}

def lang_to_dir(lang: str) -> str:
    return _LANG_TO_DIR.get(lang, lang.upper())

def dir_to_lang(d: str) -> str:
    return _DIR_TO_LANG.get(d.upper(), "pt")

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
ICONS_DIR  = ROOT.parent / "icons"
ZINES_DIR  = DATA_ROOT / "Zines"

app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path="/static")


# ── Config ─────────────────────────────────────────────────────────────────────

def load_cfg() -> dict:
    with open(CFG_PATH) as f:
        return json.load(f)

def save_cfg(cfg: dict) -> None:
    with open(CFG_PATH, "w") as f:
        json.dump(cfg, f, indent=2)

def normalize_titles(cfg: dict) -> dict:
    """Ensure series_titles[series][lang] is always a list of strings."""
    for langs in cfg.get("series_titles", {}).values():
        for l, v in list(langs.items()):
            if isinstance(v, str):
                langs[l] = [v] if v else []
            elif not isinstance(v, list):
                langs[l] = []
    return cfg

@app.route("/api/config", methods=["GET", "PUT"])
def api_config():
    if request.method == "PUT":
        save_cfg(request.json)
        return jsonify({"ok": True})
    cfg = normalize_titles(load_cfg())
    # Seed defaults only for series that have no user-configured titles yet
    st = cfg.setdefault("series_titles", {})
    for series, langs in _mt.SERIES_TITLES.items():
        for l, default_title in langs.items():
            if default_title:
                arr = st.setdefault(series, {}).setdefault(l, [])
                if not arr:
                    arr.insert(0, default_title)
    return jsonify(cfg)


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
    resp = _safe_send(DATA_ROOT / fp)
    if isinstance(resp, tuple):
        return resp
    # Zine thumbnails change on every save — never cache them
    if fp.startswith("Zines/"):
        resp.headers["Cache-Control"] = "no-store"
        resp.headers.pop("ETag", None)
        resp.headers.pop("Last-Modified", None)
    return resp

def _png_to_pdf_bytes(path: Path) -> bytes:
    return img2pdf.convert(str(path))

@app.route("/api/download/<path:fp>")
def api_download(fp):
    path = (DATA_ROOT / fp).resolve()
    if not str(path).startswith(str(DATA_ROOT.resolve())):
        return "Forbidden", 403
    if not path.exists():
        return "Not found", 404
    if path.suffix == ".pdf":
        return send_file(path, mimetype="application/pdf", as_attachment=True,
                         download_name=path.name)
    # Legacy PNG → PDF
    buf = io.BytesIO(_png_to_pdf_bytes(path))
    return send_file(buf, mimetype="application/pdf", as_attachment=True,
                     download_name=path.stem + ".pdf")

@app.route("/api/download-all")
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

@app.route("/font/<name>")
def serve_font(name):
    return _safe_send(ROOT / name)

@app.route("/icon/<name>")
def serve_icon(name):
    return _safe_send(ICONS_DIR / name)

@app.route("/api/icons")
def api_icons():
    if not ICONS_DIR.exists():
        return jsonify([])
    return jsonify([
        {"name": p.name, "url": f"icon/{p.name}"}
        for p in sorted(ICONS_DIR.glob("*.png"))
    ])


# ── Thumbnails ─────────────────────────────────────────────────────────────────

THUMBS_PATH = ROOT / "thumbnails.json"

def load_thumbnails() -> dict:
    try:
        with open(THUMBS_PATH) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

@app.route("/api/thumbnails", methods=["GET", "PUT"])
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
        "thumb":  str(thumb.relative_to(DATA_ROOT)) if thumb.exists() else None,
        "page":   None,
    }


@app.route("/api/templates", methods=["DELETE"])
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

@app.route("/api/templates")
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
            meta.update({"size_bytes": stat.st_size,
                         "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
                         "width": 0, "height": 0})
        out.append(meta)
    return jsonify(out)


# ── Final image listing ────────────────────────────────────────────────────────

@app.route("/api/images")
def api_images():
    out = []
    for path in sorted(FINAL_DIR.rglob("*.png")):
        parent = path.parent
        series = parent.name.upper() if parent != FINAL_DIR else path.stem.split("_")[0].upper()
        m      = re.match(r"^[A-Z]+_(\d+)", path.stem)
        rec = {
            "path":     str(path.relative_to(DATA_ROOT)),
            "filename": path.name,
            "series":   series,
            "index":    int(m.group(1)) if m else 0,
        }
        rec.update(img_meta(path))
        out.append(rec)
    return jsonify(out)


# ── Generation ─────────────────────────────────────────────────────────────────

@app.route("/api/generate", methods=["POST"])
def api_generate():
    data           = request.json or {}
    series         = data.get("series", "").strip().upper()
    fmt            = data.get("format", "postcard")
    lang           = data.get("lang", "pt")
    count          = data.get("count")
    icon_name      = data.get("icon")
    icon_path      = str(ICONS_DIR / icon_name) if icon_name else None
    icon_scale     = float(data.get("iconScale", 1.0))
    selected_image = data.get("selectedImage")
    forced_path    = str(DATA_ROOT / selected_image) if selected_image else None

    if not series:
        return jsonify({"ok": False, "error": "series is required"}), 400

    try:
        fmt_cfg = _mt.FORMATS.get(fmt, _mt.FORMATS["postcard"])
        size_px = fmt_cfg["size_px"]
        if count is None:
            count = len([p for p in FINAL_DIR.rglob("*.png")
                         if p.parent.name.upper() == series
                         or p.stem.split("_")[0].upper() == series])
        count = max(int(count), 1)

        canvas, layout = _mt.build_front_scattered(
            series, size_px, count,
            icon_path=icon_path, icon_scale=icon_scale, forced_path=forced_path,
        )

        layout_out = []
        for item in layout:
            p = Path(item["path"])
            try:
                url = str(p.relative_to(DATA_ROOT))
            except ValueError:
                url = str(p)
            layout_out.append({"url": url, "x": item["x"], "y": item["y"],
                               "w": item["w"], "h": item["h"]})

        L      = _mt.LAYOUT
        W, H   = size_px
        s_size = round(round(L["stamp_size"] * W) * (icon_scale if icon_name else 1.0))
        s_x1   = W - round(L["stamp_right"]  * W)
        s_y1   = H - round(L["stamp_bottom"] * H)
        stamp  = {"x": s_x1 - s_size, "y": s_y1 - s_size, "size": s_size}

        return jsonify({
            "ok":         True,
            "paths":      [],
            "previewOnly": True,
            "size_px":    list(size_px),
            "layout":     layout_out,
            "stamp":      stamp,
            "icon":       icon_name,
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/compose", methods=["POST"])
def api_compose():
    from pypdf import PdfReader, PdfWriter
    from io import BytesIO as _BytesIO

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

    if not layout or not series:
        return jsonify({"ok": False, "error": "layout and series required"}), 400

    try:
        fmt_cfg   = _mt.FORMATS.get(fmt, _mt.FORMATS["postcard"])
        size_px   = fmt_cfg["size_px"]
        W, H      = size_px
        lang_dir  = lang_to_dir(lang)
        out_dir   = TMPL_DIR / fmt.upper() / lang_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        icon_stem = icon_name.rsplit(".", 1)[0] if icon_name else None
        base_name = f"{series}_{icon_stem}" if icon_stem else series

        # Generate front
        placements = [
            {"path": str(DATA_ROOT / item["url"]),
             "x": int(item["x"]), "y": int(item["y"]),
             "w": int(item["w"]), "h": int(item["h"]),
             "rot": float(item.get("rot", 0)),
             "brightness": float(item.get("brightness", 100)),
             "contrast":   float(item.get("contrast",   100)),
             "bw": bw}
            for item in layout
        ]
        front_img = _mt.compose_layout(placements, size_px, icon_path=icon_path, icon_scale=icon_scale)

        # Generate back
        footer_text = load_cfg().get("back_footer", {}).get(lang)
        back_img = _mt.build_back(series, lang, size_px, title=custom_title, footer_text=footer_text)

        # Save custom title to config
        if custom_title:
            cfg  = normalize_titles(load_cfg())
            opts = cfg.setdefault("series_titles", {}).setdefault(series, {}).setdefault(lang, [])
            if custom_title not in opts:
                opts.append(custom_title)
            save_cfg(cfg)

        # Combine into 2-page PDF (page 1 = front, page 2 = back)
        pt_w = W / 300 * 72
        pt_h = H / 300 * 72
        page_fn = img2pdf.get_layout_fun(pagesize=(pt_w, pt_h), fit=img2pdf.FitMode.exact)
        writer = PdfWriter()
        for img_obj in (front_img, back_img):
            buf = _BytesIO()
            img_obj.convert("RGB").save(buf, "PNG", compress_level=1)
            reader = PdfReader(_BytesIO(img2pdf.convert(buf.getvalue(), layout_fun=page_fn)))
            writer.add_page(reader.pages[0])

        pdf_path   = out_dir / f"{base_name}.pdf"
        thumb_path = out_dir / f"{base_name}_thumb.png"
        out_buf = _BytesIO()
        writer.write(out_buf)
        pdf_path.write_bytes(out_buf.getvalue())

        # Thumbnail: front face at 500px wide
        thumb_h = round(H * 500 / W)
        front_img.resize((500, thumb_h), Image.LANCZOS).save(str(thumb_path), "PNG", compress_level=6)

        # Sidecar JSON for layout restoration
        L      = _mt.LAYOUT
        s_size = round(round(L["stamp_size"] * W) * (icon_scale if icon_name else 1.0))
        s_x1   = W - round(L["stamp_right"]  * W)
        s_y1   = H - round(L["stamp_bottom"] * H)
        stamp  = {"x": s_x1 - s_size, "y": s_y1 - s_size, "size": s_size}
        layout_out = [{"url": item["url"], "x": item["x"], "y": item["y"],
                       "w": item["w"], "h": item["h"], "rot": item.get("rot", 0),
                       "brightness": item.get("brightness", 100), "contrast": item.get("contrast", 100)}
                      for item in layout]
        sidecar = {"series": series, "format": fmt, "lang": lang, "icon": icon_name,
                   "iconScale": icon_scale, "bw": bw, "layout": layout_out,
                   "size_px": list(size_px), "stamp": stamp}
        (out_dir / f"{base_name}.json").write_text(json.dumps(sidecar))

        return jsonify({
            "ok":    True,
            "paths": [str(pdf_path.relative_to(DATA_ROOT))],
            "thumb": str(thumb_path.relative_to(DATA_ROOT)),
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


# ── Refresh (bulk re-render) ───────────────────────────────────────────────────

@app.route("/api/refresh-prints", methods=["POST"])
def api_refresh_prints():
    from pypdf import PdfReader, PdfWriter
    from io import BytesIO as _BytesIO

    cfg           = load_cfg()
    back_footers  = cfg.get("back_footer", {})
    raw_titles    = cfg.get("series_titles", {})
    series_titles = {s: {l: (v[0] if isinstance(v, list) else v)
                         for l, v in langs.items()}
                     for s, langs in raw_titles.items()}

    count, errors = 0, []
    for json_path in sorted(TMPL_DIR.rglob("*.json")):
        try:
            with open(json_path) as f:
                sidecar = json.load(f)
        except Exception:
            continue
        series     = sidecar.get("series", "").strip().upper()
        lang       = sidecar.get("lang", "pt")
        fmt        = sidecar.get("format", "postcard")
        icon_name  = sidecar.get("icon")
        icon_scale = float(sidecar.get("iconScale", 1.0))
        bw         = bool(sidecar.get("bw", False))
        layout     = sidecar.get("layout", [])
        size_px    = tuple(sidecar.get("size_px",
                           list(_mt.FORMATS.get(fmt, _mt.FORMATS["postcard"])["size_px"])))
        if not series or not layout:
            continue
        try:
            icon_path  = str(ICONS_DIR / icon_name) if icon_name else None
            placements = [{"path": str(DATA_ROOT / it["url"]),
                           "x": int(it["x"]), "y": int(it["y"]),
                           "w": int(it["w"]), "h": int(it["h"]),
                           "rot": float(it.get("rot", 0)),
                           "brightness": float(it.get("brightness", 100)),
                           "contrast":   float(it.get("contrast",   100)),
                           "bw": bw}
                          for it in layout]
            front_img    = _mt.compose_layout(placements, size_px, icon_path=icon_path, icon_scale=icon_scale)
            title        = series_titles.get(series, {}).get(lang)
            footer_text  = back_footers.get(lang)
            back_img     = _mt.build_back(series, lang, size_px, title=title, footer_text=footer_text)
            W, H         = size_px
            pt_w, pt_h   = W / 300 * 72, H / 300 * 72
            page_fn      = img2pdf.get_layout_fun(pagesize=(pt_w, pt_h), fit=img2pdf.FitMode.exact)
            writer       = PdfWriter()
            for img_obj in (front_img, back_img):
                buf = _BytesIO()
                img_obj.convert("RGB").save(buf, "PNG", compress_level=1)
                writer.add_page(PdfReader(_BytesIO(img2pdf.convert(buf.getvalue(), layout_fun=page_fn))).pages[0])
            out_buf = _BytesIO(); writer.write(out_buf)
            json_path.with_suffix(".pdf").write_bytes(out_buf.getvalue())
            thumb_h = round(H * 500 / W)
            front_img.resize((500, thumb_h), Image.LANCZOS).save(
                str(json_path.with_name(json_path.stem + "_thumb.png")), "PNG", compress_level=6)
            count += 1
        except Exception as e:
            errors.append(f"{json_path.stem}: {e}")
    return jsonify({"ok": True, "count": count, "errors": errors})


@app.route("/api/refresh-zines", methods=["POST"])
def api_refresh_zines():
    import math as _math

    cfg               = load_cfg()
    back_descriptions = cfg.get("back_description", {})
    raw_titles        = cfg.get("series_titles", {})
    series_titles     = {s: {l: (v[0] if isinstance(v, list) else v)
                              for l, v in langs.items()}
                         for s, langs in raw_titles.items()}

    if not ZINES_DIR.exists():
        return jsonify({"ok": True, "count": 0, "errors": []})

    count, errors = 0, []
    for json_path in sorted(ZINES_DIR.glob("*.json")):
        try:
            with open(json_path) as f:
                sidecar = json.load(f)
        except Exception:
            continue
        series    = sidecar.get("series", "").strip().upper()
        lang      = sidecar.get("lang", "pt")
        fmt       = sidecar.get("format", "mini-poster")
        layout    = sidecar.get("layout", [])
        canvas_w  = int(sidecar.get("canvasW", 680))
        canvas_h  = int(sidecar.get("canvasH", 380))
        bw        = bool(sidecar.get("bw", False))
        texts     = [{"left": {**t["left"]}, "right": {**t["right"]}}
                     for t in sidecar.get("texts", [])]
        icon_name = sidecar.get("icon")
        if not series or not layout:
            continue

        HALF_W = canvas_w // 2

        # Trim/pad layout: odd M → 3 trailing blanks (N-2, N-1, N);
        # even M → 4 trailing blanks (N-3, N-2, N-1, N).
        # Round up to nearest multiple of 4 so imposition adds no hidden extra blanks.
        last_img_idx = 1
        for i, item in enumerate(layout):
            if item.get("url"):
                last_img_idx = i
        M_count  = last_img_idx - 1  # content images (excludes cover at index 0)
        base_len = (last_img_idx + 5) if M_count % 2 == 0 else (last_img_idx + 4)
        desired_len = ((base_len + 3) // 4) * 4
        if len(layout) > desired_len:
            layout = layout[:desired_len]
        while len(layout) < desired_len:
            ai = len(layout)
            layout.append({"url": None, "path": None,
                            "x": (ai % 2) * HALF_W, "y": 0,
                            "w": HALF_W, "h": canvas_h,
                            "rot": 0, "brightness": 100, "contrast": 100, "white": True})

        # Ensure texts array is long enough
        num_spreads = _math.ceil(len(layout) / 2)

        while len(texts) < num_spreads:
            texts.append({"left":  {"text": "", "x": 0,      "y": canvas_h // 2 - 12},
                          "right": {"text": "", "x": HALF_W, "y": 12}})

        # Update cover spread texts from current config
        title_text = series_titles.get(series, {}).get(lang, "")
        if not title_text:
            for fl in ("pt", "en", "es", "fr"):
                title_text = series_titles.get(series, {}).get(fl, "")
                if title_text:
                    break
        bd_text = (back_descriptions.get(lang)
                   or back_descriptions.get("pt")
                   or back_descriptions.get("en") or "")

        _BD_MARGIN = 28  # must match CANVAS_MARGIN in JS and _make_zine_pdf
        if title_text:
            texts[0]["left"]["text"] = title_text.strip().upper()
        if bd_text:
            texts[0]["right"]["text"] = bd_text.strip().upper()
        # Always reset cover positions to canonical values so padding updates take effect
        if texts[0]["left"].get("x") is None:
            texts[0]["left"]["x"] = 0
        if texts[0]["left"].get("y") is None:
            texts[0]["left"]["y"] = canvas_h // 2 - 12
        texts[0]["right"]["x"] = HALF_W + _BD_MARGIN
        texts[0]["right"]["y"] = 24

        try:
            icon_path = str(ICONS_DIR / icon_name) if icon_name else None
            pdf_bytes, thumb_bytes = _make_zine_pdf(
                layout, canvas_w, canvas_h, bw, texts, fmt, icon_path=icon_path)
            (ZINES_DIR / f"{json_path.stem}.pdf").write_bytes(pdf_bytes)
            if thumb_bytes:
                (ZINES_DIR / f"{json_path.stem}_thumb.png").write_bytes(thumb_bytes)
            sidecar["texts"]  = texts
            sidecar["layout"] = layout
            json_path.write_text(json.dumps(sidecar, ensure_ascii=False))
            count += 1
        except Exception as e:
            errors.append(f"{json_path.stem}: {e}")
    return jsonify({"ok": True, "count": count, "errors": errors})


# ── Layout sidecar ─────────────────────────────────────────────────────────────

@app.route("/api/layout-data/<path:fp>")
def api_layout_data(fp):
    path     = (DATA_ROOT / fp).with_suffix(".json").resolve()
    tmpl_res = TMPL_DIR.resolve()
    if not str(path).startswith(str(tmpl_res)):
        return "Forbidden", 403
    if not path.exists():
        return jsonify(None)
    return jsonify(json.loads(path.read_text()))


# ── Zine CRUD ──────────────────────────────────────────────────────────────────

@app.route("/api/zines")
def api_zines():
    if not ZINES_DIR.exists():
        resp = jsonify([])
        resp.headers["Cache-Control"] = "no-store"
        return resp
    result = []
    for pdf in sorted(ZINES_DIR.glob("*.pdf"), key=lambda p: p.stat().st_mtime, reverse=True):
        meta_path = pdf.with_suffix(".json")
        meta = {}
        if meta_path.exists():
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
            except Exception:
                pass
        stat = pdf.stat()
        result.append({
            "name":       pdf.stem,
            "path":       str(pdf.relative_to(DATA_ROOT)),
            "series":     meta.get("series", ""),
            "cover":      meta.get("cover", ""),
            "layout":     meta.get("layout", []),
            "canvasW":    meta.get("canvasW", 680),
            "canvasH":    meta.get("canvasH", 380),
            "format":     meta.get("format", "postcard"),
            "lang":       meta.get("lang", ""),
            "bw":         meta.get("bw", False),
            "texts":      meta.get("texts", []),
            "icon":        meta.get("icon"),
            "globalScale": meta.get("globalScale", 100),
            "coverScale":  meta.get("coverScale",  100),
            "size_bytes": stat.st_size,
            "modified":   datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
        })
    resp = jsonify(result)
    resp.headers["Cache-Control"] = "no-store"
    return resp


@app.route("/api/zine/create", methods=["POST"])
def api_zine_create():
    data     = request.json or {}
    series   = data.get("series", "").upper()
    fmt      = data.get("format", "postcard")
    layout   = data.get("layout", [])
    canvas_w = int(data.get("canvasW", 680))
    canvas_h = int(data.get("canvasH", 380))
    bw        = bool(data.get("bw", False))
    lang      = data.get("lang", "pt") or "pt"
    texts     = data.get("texts", [])
    icon_name = (data.get("icon") or "").strip() or None

    if not series:
        return jsonify({"error": "no series"}), 400

    # White-page items have null path; require at least one real image
    if not any(it.get("path") for it in layout):
        return jsonify({"error": "no images in layout"}), 400

    data_root_str = str(DATA_ROOT.resolve())
    for item in layout:
        if not item.get("path"):
            continue  # mandatory white page — no file to validate
        p = (DATA_ROOT / item["path"]).resolve()
        if not str(p).startswith(data_root_str):
            return jsonify({"error": "invalid path"}), 400
        if not p.exists():
            return jsonify({"error": f"not found: {item['path']}"}), 400

    try:
        icon_path = str(ICONS_DIR / icon_name) if icon_name else None
        booklet, thumb_bytes = _make_zine_pdf(layout, canvas_w, canvas_h, bw, texts, fmt,
                                              icon_path=icon_path)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    ZINES_DIR.mkdir(parents=True, exist_ok=True)
    existing_name = (data.get("name") or "").strip()
    if existing_name and (ZINES_DIR / f"{existing_name}.pdf").exists():
        name = existing_name
    else:
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"{series}_{ts}"
    pdf_path = ZINES_DIR / f"{name}.pdf"
    pdf_path.write_bytes(booklet)

    if thumb_bytes:
        thumb_file = ZINES_DIR / f"{name}_thumb.png"
        thumb_file.write_bytes(thumb_bytes)
        cover_path = str(thumb_file.relative_to(DATA_ROOT))
    else:
        cover_path = layout[0]["path"]

    global_scale = int(data.get("globalScale") or 100)
    cover_scale  = int(data.get("coverScale")  or 100)
    sidecar = {"series": series, "format": fmt, "lang": lang, "cover": cover_path,
               "layout": layout, "canvasW": canvas_w, "canvasH": canvas_h,
               "bw": bw, "texts": texts, "icon": icon_name,
               "globalScale": global_scale, "coverScale": cover_scale}
    (ZINES_DIR / f"{name}.json").write_text(json.dumps(sidecar, indent=2))

    stat = pdf_path.stat()
    return jsonify({
        "ok":         True,
        "name":       name,
        "path":       str(pdf_path.relative_to(DATA_ROOT)),
        "series":     series,
        "cover":      cover_path,
        "size_bytes": stat.st_size,
        "modified":   datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
    })


@app.route("/api/zines/delete", methods=["DELETE"])
def api_zines_delete():
    names     = (request.json or {}).get("names", [])
    zines_str = str(ZINES_DIR.resolve())
    deleted   = 0
    for name in names:
        if "/" in name or "\\" in name or ".." in name:
            continue
        for ext in (".pdf", ".json", "_thumb.png"):
            p = (ZINES_DIR / (name + ext)).resolve()
            if str(p).startswith(zines_str) and p.exists():
                p.unlink()
                deleted += 1
    return jsonify({"ok": True, "deleted": deleted})


@app.route("/api/zines/download-all")
def api_zines_download_all():
    if not ZINES_DIR.exists():
        return jsonify({"error": "No zines found"}), 404
    pdfs = sorted(ZINES_DIR.glob("*.pdf"))
    if not pdfs:
        return jsonify({"error": "No zines found"}), 404
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for pdf in pdfs:
            zf.write(pdf, pdf.name)
    buf.seek(0)
    return send_file(buf, mimetype="application/zip", as_attachment=True,
                     download_name="DE_BRUITS_ZINES.zip")


@app.route("/api/zines/download/<name>")
def api_zine_download(name):
    if "/" in name or "\\" in name or ".." in name:
        return "Forbidden", 403
    pdf_path = ZINES_DIR / f"{name}.pdf"
    resolved = pdf_path.resolve()
    if not str(resolved).startswith(str(ZINES_DIR.resolve())):
        return "Forbidden", 403
    if not resolved.exists():
        return "Not found", 404
    resp = send_file(resolved, as_attachment=True, download_name=f"{name}.pdf")
    resp.headers["Cache-Control"] = "no-store"
    resp.headers.pop("ETag", None)
    resp.headers.pop("Last-Modified", None)
    return resp

def _fit_and_rotate(img, w, h, rot):
    """Fit img into w×h (object-fit: contain), then rotate in-place (no expand)."""
    ratio = min(w / img.width, h / img.height)
    fw = max(1, round(img.width  * ratio))
    fh = max(1, round(img.height * ratio))
    resized = img.resize((fw, fh), Image.LANCZOS)
    cell = Image.new("RGB", (w, h), (255, 255, 255))
    cell.paste(resized, ((w - fw) // 2, (h - fh) // 2))
    if rot:
        cell = cell.rotate(-rot, expand=False, resample=Image.BICUBIC, fillcolor=(255, 255, 255))
    return cell


ZINE_DPI = 300
# Each PDF page is a two-page spread (2 pages side by side).
# mini-poster pages are A5 → spread = A4 landscape (297 × 210 mm)
# poster pages are A4      → spread = A3 landscape (420 × 297 mm)
ZINE_PAPER_SIZES = {
    'mini-poster': (297 / 25.4, 210 / 25.4),   # A4 landscape in inches
    'poster':      (420 / 25.4, 297 / 25.4),   # A3 landscape in inches
}

def _make_zine_pdf(layout, canvas_w, canvas_h, bw=False, texts=None, fmt='mini-poster',
                   icon_path=None):
    from pypdf import PdfReader, PdfWriter
    from PIL import ImageEnhance, ImageDraw, ImageFont
    from io import BytesIO
    import numpy as np

    if not layout:
        raise ValueError("empty layout")

    page_w_in, page_h_in = ZINE_PAPER_SIZES.get(fmt, ZINE_PAPER_SIZES['mini-poster'])
    spread_w = round(page_w_in * ZINE_DPI)
    spread_h = round(page_h_in * ZINE_DPI)
    scale    = spread_w / canvas_w

    page_layout_fn = img2pdf.get_layout_fun(
        pagesize=(page_w_in * 72, page_h_in * 72),
        fit=img2pdf.FitMode.exact,
    )

    font_path = Path(__file__).parent / "DebruitsRegular-Handwritten.ttf"
    try:
        text_font = ImageFont.truetype(str(font_path), max(12, round(28 * scale)))
        desc_font = ImageFont.truetype(str(font_path), max(8,  round(18 * scale)))
        num_font  = ImageFont.truetype(str(font_path), max(10, round(22 * scale)))
    except Exception:
        text_font = ImageFont.load_default()
        desc_font = text_font
        num_font  = text_font

    n       = len(layout)
    half_cw = canvas_w // 2

    # Pad reading-order pages to a multiple of 4 so the booklet folds evenly.
    total_pages = ((n + 3) // 4) * 4
    n_sheets    = total_pages // 4

    # Booklet imposition: print odd PDF pages (fronts), flip stack, print even PDF pages (backs).
    # Sheet k (outermost = 1):
    #   front (PDF page 2k-1): left = reading-page(N-2k+2), right = reading-page(2k-1)
    #   back  (PDF page 2k  ): left = reading-page(2k),     right = reading-page(N-2k+1)
    # Stored as 0-based layout indices.
    imposition = []
    for k in range(1, n_sheets + 1):
        imposition.append((total_pages - 2*k + 1, 2*k - 2))   # front
        imposition.append((2*k - 1,               total_pages - 2*k))  # back

    def load_img(pg):
        if pg < 0 or pg >= n:
            return None
        item = layout[pg]
        if not item.get("path"):
            return None  # mandatory white page
        raw = Image.open(str((DATA_ROOT / item["path"]).resolve()))
        # Composite against white first so PIL doesn't blend against black (its
        # default for RGBA→RGB), which would darken semi-transparent edges vs browser.
        if raw.mode in ("RGBA", "LA", "P"):
            rgba = raw.convert("RGBA")
            r, g, b_ch, a = rgba.split()
            bg_white = Image.new("RGB", raw.size, (255, 255, 255))
            bg_white.paste(Image.merge("RGB", (r, g, b_ch)), mask=a)
            return bg_white
        else:
            return raw.convert("RGB")

    def place_x(pg, target_right):
        # Convert stored canvas-x (in original half) to position on target half of new spread.
        item = layout[pg]
        orig_right = (pg % 2 == 1)
        local_x = item["x"] - (half_cw if orig_right else 0)
        return round((local_x + (half_cw if target_right else 0)) * scale)

    writer      = PdfWriter()
    thumb_bytes = None

    for pdf_idx, (left_pg, right_pg) in enumerate(imposition):
        spread = Image.new("RGB", (spread_w, spread_h), (255, 255, 255))

        for target_right, pg in ((False, left_pg), (True, right_pg)):
            img = load_img(pg)
            if img is None:
                continue
            item = layout[pg]
            x_a  = place_x(pg, target_right)
            y_a  = round(item["y"] * scale)
            w_a  = max(1, round(item["w"] * scale))
            h_a  = max(1, round(item["h"] * scale))
            # Fit first (scale to display size), then apply effects — matches browser
            # CSS filter behaviour: filters are applied at rendering resolution, not
            # at source resolution.  Applying contrast before downscale would clip
            # highs/lows first and then average them back toward grey, washing out
            # the effect; browsers clip after averaging, producing crisper contrast.
            fitted = _fit_and_rotate(img, w_a, h_a, float(item.get("rot", 0)))
            bval = float(item.get("brightness", 100))
            cval = float(item.get("contrast",   100))
            if bval != 100:
                fitted = ImageEnhance.Brightness(fitted).enhance(bval / 100.0)
            if cval != 100:
                arr = np.array(fitted, dtype=np.float32)
                arr = (arr - 128.0) * (cval / 100.0) + 128.0
                fitted = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
            if bw or bool(item.get("bw", False)):
                fitted = fitted.convert("L").convert("RGB")
            spread.paste(fitted, (x_a, y_a))

        draw = ImageDraw.Draw(spread)

        # Text labels — positions are stored in canvas pixels; scale directly to PDF pixels.
        # CANVAS_MARGIN (28 canvas-px) is baked into the auto-placed cover texts so both
        # the preview and the PDF show identical margins without any extra offset here.
        CANVAS_MARGIN = 28
        for target_right, pg in ((False, left_pg), (True, right_pg)):
            if pg < 0 or pg >= n:
                continue
            si  = pg // 2
            sid = 'right' if pg % 2 == 1 else 'left'
            ti  = (texts or [])[si].get(sid, {}) if si < len(texts or []) else {}
            txt = ti.get("text", "")
            if not txt:
                continue
            orig_right = (pg % 2 == 1)
            local_tx   = ti.get("x", 0) - (half_cw if orig_right else 0)
            ty         = round(ti.get("y", 0) * scale)
            box_start  = round(local_tx * scale)
            page_off   = round(half_cw * scale) if target_right else 0
            tx         = page_off + box_start
            is_back_desc = (si == 0 and sid == 'right')
            # back-desc has CANVAS_MARGIN baked in on the left; apply the matching right margin
            box_w = (round((half_cw - 2 * CANVAS_MARGIN) * scale) if is_back_desc
                     else round(half_cw * scale))
            if is_back_desc:
                # Justified word-wrap for cover back description (smaller font)
                try:
                    lh = round(draw.textbbox((0, 0), "A", font=desc_font)[3] * 1.4)
                except Exception:
                    lh = round(18 * scale * 1.4)
                cy = ty
                for para in txt.split('\n\n'):
                    para = para.strip()
                    if not para:
                        cy += lh; continue
                    words = para.split()
                    lines, line = [], []
                    for word in words:
                        test = ' '.join(line + [word])
                        try:
                            w = draw.textlength(test, font=desc_font)
                        except Exception:
                            w = len(test) * 8
                        if w <= box_w or not line:
                            line.append(word)
                        else:
                            lines.append(line); line = [word]
                    if line:
                        lines.append(line)
                    for li, ln in enumerate(lines):
                        is_last = (li == len(lines) - 1)
                        if is_last or len(ln) == 1:
                            draw.text((tx, cy), ' '.join(ln), fill=(0, 0, 0), font=desc_font)
                        else:
                            try:
                                total_w = sum(draw.textlength(w, font=desc_font) for w in ln)
                                gap = (box_w - total_w) / (len(ln) - 1)
                            except Exception:
                                gap = 8
                            cx = tx
                            for word in ln:
                                draw.text((round(cx), cy), word, fill=(0, 0, 0), font=desc_font)
                                try:
                                    cx += draw.textlength(word, font=desc_font) + gap
                                except Exception:
                                    cx += len(word) * 8 + gap
                        cy += lh
                    cy += lh
            else:
                # Centered — handle multiline titles by centering each line
                is_title = (si == 0 and sid == 'left')
                stroke   = max(1, round(scale * 0.3)) if is_title else 0
                if is_title:
                    txt = _auto_split_title(txt)
                lines_txt = txt.split('\n')
                try:
                    lh_title = round(draw.textbbox((0, 0), "A", font=text_font)[3] * 1.2)
                except Exception:
                    lh_title = round(28 * scale * 1.2)
                cy = ty
                for ln in lines_txt:
                    tb = draw.textbbox((0, 0), ln, font=text_font)
                    lw = tb[2] - tb[0]
                    draw.text((tx + (box_w - lw) // 2, cy), ln, fill=(0, 0, 0), font=text_font,
                              stroke_width=stroke, stroke_fill=(0, 0, 0))
                    cy += lh_title

        # Page numbers — skip cover/inside-cover, inside-back/back-cover, and filler pages
        margin = round(16 * scale)
        num_y  = spread_h - margin - round(22 * scale)
        for target_right, pg in ((False, left_pg), (True, right_pg)):
            if pg < 2 or pg >= n - 2:
                continue
            if pg >= n or not layout[pg].get("path"):
                continue  # filler/blank page — no number
            page_num = pg - 1
            if target_right:
                bbox = draw.textbbox((0, 0), str(page_num), font=num_font)
                draw.text((spread_w - margin - (bbox[2] - bbox[0]), num_y),
                          str(page_num), fill=(80, 80, 80), font=num_font)
            else:
                draw.text((margin, num_y), str(page_num), fill=(80, 80, 80), font=num_font)

        # Icon stamp on front cover (layout[0] = pg 0), bottom-right — same as prints
        if icon_path and 0 < n:
            stamp_side = None
            if right_pg == 0:
                stamp_side = 'right'
            elif left_pg == 0:
                stamp_side = 'left'
            if stamp_side:
                x_off    = spread_w // 2 if stamp_side == 'right' else 0
                half_img = spread.crop((x_off, 0, x_off + spread_w // 2, spread_h)).convert("RGBA")
                _mt.place_stamp(half_img, spread_w // 2, spread_h,
                                icon_path=icon_path, icon_scale=2.0)
                spread.paste(half_img.convert("RGB"), (x_off, 0))

        # Thumbnail: right half of first PDF page = reading page 1 (front cover)
        if pdf_idx == 0:
            page1       = spread.crop((spread_w // 2, 0, spread_w, spread_h))
            thumb_w     = 500
            thumb_h     = round(spread_h * thumb_w / (spread_w // 2))
            page1_thumb = page1.resize((thumb_w, thumb_h), Image.LANCZOS)

            # Re-render the cover image at thumbnail scale so effects (brightness/
            # contrast) are applied at display resolution — matching how the browser
            # applies CSS filters (after scaling, not before).  Without this, heavy
            # contrast settings look much weaker in the thumbnail than in the canvas
            # preview, because clipping at high-res then downsampling softens the
            # effect compared to downsampling then clipping.
            t_scale = thumb_w / (canvas_w // 2)   # ≈ 1.47 — thumbnail px per canvas px
            t_pg    = right_pg                     # front cover = right side of spread 0
            if 0 <= t_pg < n:
                t_item = layout[t_pg]
                if t_item.get("path"):
                    t_raw = load_img(t_pg)         # composited against white, no effects
                    # local-x for the right side: even pages use item["x"] directly
                    t_lx  = t_item["x"] - (half_cw if t_pg % 2 == 1 else 0)
                    t_x   = round(t_lx  * t_scale)
                    t_y   = round(t_item["y"] * t_scale)
                    t_w   = max(1, round(t_item["w"] * t_scale))
                    t_h   = max(1, round(t_item["h"] * t_scale))
                    # Fit image (object-fit:contain) without wrapping in a white cell —
                    # wrapping with white would overwrite text labels that fall inside the
                    # cell bounding box.  Instead compute the fitted size and paste only
                    # the image content at the centred position within the cell.
                    t_ratio = min(t_w / t_raw.width, t_h / t_raw.height)
                    t_fw    = max(1, round(t_raw.width  * t_ratio))
                    t_fh    = max(1, round(t_raw.height * t_ratio))
                    t_img   = t_raw.resize((t_fw, t_fh), Image.LANCZOS)
                    t_rot   = float(t_item.get("rot", 0))
                    if t_rot:
                        t_img = t_img.rotate(-t_rot, expand=False,
                                             resample=Image.BICUBIC,
                                             fillcolor=(255, 255, 255))
                    t_bv  = float(t_item.get("brightness", 100))
                    t_cv  = float(t_item.get("contrast",   100))
                    if t_bv != 100:
                        t_img = ImageEnhance.Brightness(t_img).enhance(t_bv / 100.0)
                    if t_cv != 100:
                        t_arr = np.array(t_img, dtype=np.float32)
                        t_arr = (t_arr - 128.0) * (t_cv / 100.0) + 128.0
                        t_img = Image.fromarray(np.clip(t_arr, 0, 255).astype(np.uint8))
                    if bw or bool(t_item.get("bw", False)):
                        t_img = t_img.convert("L").convert("RGB")
                    page1_thumb.paste(t_img,
                                      (t_x + (t_w - t_fw) // 2,
                                       t_y + (t_h - t_fh) // 2))

            tb          = io.BytesIO()
            page1_thumb.save(tb, "PNG", compress_level=6)
            thumb_bytes = tb.getvalue()

        buf = io.BytesIO()
        spread.save(buf, "PNG", compress_level=1)
        reader = PdfReader(BytesIO(img2pdf.convert(buf.getvalue(), layout_fun=page_layout_fn)))
        writer.add_page(reader.pages[0])

    out_buf = BytesIO()
    writer.write(out_buf)
    return out_buf.getvalue(), thumb_bytes


# ── Entry point ────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_file(STATIC_DIR / "index.html")

if __name__ == "__main__":
    print(f"\nDE BRUITS dashboard → http://localhost:5000")
    print(f"  Data root : {DATA_ROOT}")
    print()
    app.run(debug=False, port=5000)

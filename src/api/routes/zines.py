"""Zine CRUD routes and PDF imposition engine."""

import io
import json
import math
import zipfile
from datetime import datetime

import img2pdf
from flask import Blueprint, jsonify, request, send_file
from PIL import Image
from pypdf import PdfReader, PdfWriter

from .. import compositor as _mt
from ..paths import DATA_ROOT, FONTS_DIR, ICONS_DIR, STAMP_EXTRA_BOTTOM, ZINES_DIR
from ..utils import auto_split_title, fit_and_rotate, fit_and_rotate_rgba, flatten_titles, load_cfg

bp = Blueprint("zines", __name__)

ZINE_DPI = 300
ZINE_PAPER_SIZES = {
    "mini-poster": (297 / 25.4, 210 / 25.4),
    "poster":      (420 / 25.4, 297 / 25.4),
}


# ── PDF imposition engine ──────────────────────────────────────────────────────

def _make_zine_pdf(layout, canvas_w, canvas_h, bw=False, texts=None, fmt="mini-poster",
                   icon_path=None):
    from PIL import ImageDraw, ImageFont

    if not layout:
        raise ValueError("empty layout")

    page_w_in, page_h_in = ZINE_PAPER_SIZES.get(fmt, ZINE_PAPER_SIZES["mini-poster"])
    spread_w = round(page_w_in * ZINE_DPI)
    spread_h = round(page_h_in * ZINE_DPI)
    scale    = spread_w / canvas_w

    page_layout_fn = img2pdf.get_layout_fun(
        pagesize=(page_w_in * 72, page_h_in * 72),
        fit=img2pdf.FitMode.exact,
    )

    font_path = FONTS_DIR / "DebruitsRegular-Handwritten.ttf"
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

    total_pages = ((n + 3) // 4) * 4
    n_sheets    = total_pages // 4

    # Booklet imposition: sheet k (1-based, outermost first)
    #   front spread: left = page N-2k+2, right = page 2k-1
    #   back  spread: left = page 2k,     right = page N-2k+1
    imposition = []
    for k in range(1, n_sheets + 1):
        imposition.append((total_pages - 2*k + 1, 2*k - 2))
        imposition.append((2*k - 1,               total_pages - 2*k))

    def _load_raw(path_str):
        raw = Image.open(str((DATA_ROOT / path_str).resolve()))
        if raw.mode in ("RGBA", "LA", "P"):
            rgba = raw.convert("RGBA")
            r, g, b_ch, a = rgba.split()
            a = a.point(lambda v: 255 if v > 127 else 0)
            bg = Image.new("RGB", raw.size, (255, 255, 255))
            bg.paste(Image.merge("RGB", (r, g, b_ch)), mask=a)
            return bg
        return raw.convert("RGB")

    def _render_sub(sub_item, pg, target_right, spread):
        path_str = sub_item.get("path")
        if not path_str:
            return
        bval  = float(sub_item.get("brightness", 100))
        cval  = float(sub_item.get("contrast",   100))
        do_bw = bw or bool(sub_item.get("bw", False))
        try:
            raw = Image.open(str((DATA_ROOT / path_str).resolve()))
        except Exception:
            return
        orig_right = (pg % 2 == 1)
        local_x    = sub_item["x"] - (half_cw if orig_right else 0)
        x_a = round((local_x + (half_cw if target_right else 0)) * scale)
        y_a = max(0, round(sub_item["y"] * scale))
        w_a = max(1, round(sub_item["w"] * scale))
        h_a = max(1, round(sub_item["h"] * scale))
        rot = float(sub_item.get("rot", 0))
        if raw.mode in ("RGBA", "LA", "P"):
            rgba = raw.convert("RGBA")
            r, g, b_ch, a = rgba.split()
            a = a.point(lambda v: 255 if v > 127 else 0)
            content = _mt.apply_bc(Image.merge("RGB", (r, g, b_ch)), bval, cval, do_bw)
            cr, cg, cb = content.split()
            fitted = fit_and_rotate_rgba(Image.merge("RGBA", (cr, cg, cb, a)), w_a, h_a, rot)
            fr, fg, fb, fa = fitted.split()
            spread.paste(Image.merge("RGB", (fr, fg, fb)), (x_a, y_a), mask=fa)
        else:
            spread.paste(fit_and_rotate(_mt.apply_bc(raw.convert("RGB"), bval, cval, do_bw),
                                        w_a, h_a, rot), (x_a, y_a))

    writer      = PdfWriter()
    thumb_bytes = None

    for pdf_idx, (left_pg, right_pg) in enumerate(imposition):
        spread = Image.new("RGB", (spread_w, spread_h), (255, 255, 255))

        for target_right, pg in ((False, left_pg), (True, right_pg)):
            if pg < 0 or pg >= n:
                continue
            item = layout[pg]
            subs = item.get("images")
            if subs:
                for sub in subs:
                    _render_sub(sub, pg, target_right, spread)
            elif item.get("path"):
                _render_sub(item, pg, target_right, spread)

        draw = ImageDraw.Draw(spread)

        # CANVAS_MARGIN must match the constant in app.js and in the cover text positions.
        CANVAS_MARGIN = 28
        for target_right, pg in ((False, left_pg), (True, right_pg)):
            if pg < 0 or pg >= n:
                continue
            si  = pg // 2
            sid = "right" if pg % 2 == 1 else "left"
            ti  = (texts or [])[si].get(sid, {}) if si < len(texts or []) else {}
            txt = ti.get("text", "")
            if not txt:
                continue
            orig_right = (pg % 2 == 1)
            local_tx   = ti.get("x", 0) - (half_cw if orig_right else 0)
            ty         = round(ti.get("y", 0) * scale)
            tx         = (round(half_cw * scale) if target_right else 0) + round(local_tx * scale)
            is_back_desc = (si == 0 and sid == "right")
            box_w = (round((half_cw - 2 * CANVAS_MARGIN) * scale) if is_back_desc
                     else round(half_cw * scale))
            if is_back_desc:
                try:
                    lh = round(draw.textbbox((0, 0), "A", font=desc_font)[3] * 1.4)
                except Exception:
                    lh = round(18 * scale * 1.4)
                cy = ty
                for para in txt.split("\n\n"):
                    para = para.strip()
                    if not para:
                        cy += lh; continue
                    words = para.split()
                    lines, line = [], []
                    for word in words:
                        test = " ".join(line + [word])
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
                            draw.text((tx, cy), " ".join(ln), fill=(0, 0, 0), font=desc_font)
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
                is_title = (si == 0 and sid == "left")
                stroke   = max(1, round(scale * 0.3)) if is_title else 0
                if is_title and "\n" not in txt:
                    txt = auto_split_title(txt)
                lines_txt = txt.split("\n")
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

        margin = round(16 * scale)
        num_y  = spread_h - margin - round(22 * scale)
        for target_right, pg in ((False, left_pg), (True, right_pg)):
            if pg < 2 or pg >= n - 2:
                continue
            if not layout[pg].get("path"):
                continue
            page_num = pg - 1
            if target_right:
                bbox = draw.textbbox((0, 0), str(page_num), font=num_font)
                draw.text((spread_w - margin - (bbox[2] - bbox[0]), num_y),
                          str(page_num), fill=(80, 80, 80), font=num_font)
            else:
                draw.text((margin, num_y), str(page_num), fill=(80, 80, 80), font=num_font)

        if icon_path and n > 0:
            stamp_side = ("right" if right_pg == 0 else "left" if left_pg == 0 else None)
            if stamp_side:
                x_off    = spread_w // 2 if stamp_side == "right" else 0
                half_img = spread.crop((x_off, 0, x_off + spread_w // 2, spread_h)).convert("RGBA")
                _mt.place_stamp(half_img, spread_w // 2, spread_h,
                                icon_path=icon_path, icon_scale=2.0,
                                extra_bottom=STAMP_EXTRA_BOTTOM)
                spread.paste(half_img.convert("RGB"), (x_off, 0))

        if pdf_idx == 0:
            thumb_w     = 500
            thumb_h     = round(spread_h * thumb_w / (spread_w // 2))
            page1_thumb = spread.crop((spread_w // 2, 0, spread_w, spread_h)).resize(
                (thumb_w, thumb_h), Image.LANCZOS)

            t_scale = thumb_w / (canvas_w // 2)
            t_pg    = right_pg
            if 0 <= t_pg < n:
                t_item = layout[t_pg]
                t_src  = (t_item.get("images") or [None])[0] or t_item
                if t_src.get("path"):
                    try:
                        t_raw   = _load_raw(t_src["path"])
                        t_lx    = t_src["x"] - (half_cw if t_pg % 2 == 1 else 0)
                        t_x     = round(t_lx  * t_scale)
                        t_y     = round(t_src["y"] * t_scale)
                        t_w     = max(1, round(t_src["w"] * t_scale))
                        t_h     = max(1, round(t_src["h"] * t_scale))
                        t_ratio = min(t_w / t_raw.width, t_h / t_raw.height)
                        t_fw    = max(1, round(t_raw.width  * t_ratio))
                        t_fh    = max(1, round(t_raw.height * t_ratio))
                        t_img   = t_raw.resize((t_fw, t_fh), Image.LANCZOS)
                        if float(t_src.get("rot", 0)):
                            t_img = t_img.rotate(-float(t_src["rot"]), expand=False,
                                                 resample=Image.BICUBIC,
                                                 fillcolor=(255, 255, 255))
                        t_img = _mt.apply_bc(t_img,
                                             float(t_src.get("brightness", 100)),
                                             float(t_src.get("contrast",   100)),
                                             bw or bool(t_src.get("bw", False)))
                        page1_thumb.paste(t_img, (t_x + (t_w - t_fw) // 2,
                                                   t_y + (t_h - t_fh) // 2))
                    except Exception:
                        pass

            tb          = io.BytesIO()
            page1_thumb.save(tb, "PNG", compress_level=6)
            thumb_bytes = tb.getvalue()

        buf = io.BytesIO()
        spread.save(buf, "PNG", compress_level=1)
        writer.add_page(
            PdfReader(io.BytesIO(img2pdf.convert(buf.getvalue(),
                                                 layout_fun=page_layout_fn))).pages[0])

    out_buf = io.BytesIO()
    writer.write(out_buf)
    return out_buf.getvalue(), thumb_bytes


# ── Routes ─────────────────────────────────────────────────────────────────────

@bp.route("/api/zines")
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
            "name":        pdf.stem,
            "path":        str(pdf.relative_to(DATA_ROOT)),
            "series":      meta.get("series", ""),
            "cover":       meta.get("cover", ""),
            "layout":      meta.get("layout", []),
            "canvasW":     meta.get("canvasW", 680),
            "canvasH":     meta.get("canvasH", 380),
            "format":      meta.get("format", "postcard"),
            "lang":        meta.get("lang", ""),
            "bw":          meta.get("bw", False),
            "texts":       meta.get("texts", []),
            "icon":        meta.get("icon"),
            "globalScale": meta.get("globalScale", 100),
            "coverScale":  meta.get("coverScale",  100),
            "M":           meta.get("M", 1),
            "size_bytes":  stat.st_size,
            "modified":    datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
        })
    resp = jsonify(result)
    resp.headers["Cache-Control"] = "no-store"
    return resp


@bp.route("/api/zine/create", methods=["POST"])
def api_zine_create():
    data      = request.json or {}
    series    = data.get("series", "").upper()
    fmt       = data.get("format", "postcard")
    layout    = data.get("layout", [])
    canvas_w  = int(data.get("canvasW", 680))
    canvas_h  = int(data.get("canvasH", 380))
    bw        = bool(data.get("bw", False))
    lang      = data.get("lang", "pt") or "pt"
    texts     = data.get("texts", [])
    icon_name = (data.get("icon") or "").strip() or None

    if not series:
        return jsonify({"error": "no series"}), 400

    def _has_real_img(it):
        return bool((it.get("images") and any(s.get("path") for s in it["images"]))
                    or it.get("path"))

    if not any(_has_real_img(it) for it in layout):
        return jsonify({"error": "no images in layout"}), 400

    data_root_str = str(DATA_ROOT.resolve())
    for item in layout:
        paths_to_check = [sub["path"] for sub in item.get("images", []) if sub.get("path")]
        if item.get("path") and not item.get("images"):
            paths_to_check.append(item["path"])
        for rel_path in paths_to_check:
            p = (DATA_ROOT / rel_path).resolve()
            if not str(p).startswith(data_root_str):
                return jsonify({"error": "invalid path"}), 400
            if not p.exists():
                return jsonify({"error": f"not found: {rel_path}"}), 400

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
        name = f"{series}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    pdf_path = ZINES_DIR / f"{name}.pdf"
    pdf_path.write_bytes(booklet)

    if thumb_bytes:
        thumb_file = ZINES_DIR / f"{name}_thumb.png"
        thumb_file.write_bytes(thumb_bytes)
        cover_path = str(thumb_file.relative_to(DATA_ROOT))
    else:
        cover_path = layout[0]["path"]

    sidecar = {
        "series": series, "format": fmt, "lang": lang, "cover": cover_path,
        "layout": layout, "canvasW": canvas_w, "canvasH": canvas_h,
        "bw": bw, "texts": texts, "icon": icon_name,
        "globalScale": int(data.get("globalScale") or 100),
        "coverScale":  int(data.get("coverScale")  or 100),
        "M":           int(data.get("M") or 1),
    }
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


@bp.route("/api/zines/delete", methods=["DELETE"])
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


@bp.route("/api/zines/download-all")
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


@bp.route("/api/zines/download/<name>")
def api_zine_download(name):
    if "/" in name or "\\" in name or ".." in name:
        return "Forbidden", 403
    resolved = (ZINES_DIR / f"{name}.pdf").resolve()
    if not str(resolved).startswith(str(ZINES_DIR.resolve())):
        return "Forbidden", 403
    if not resolved.exists():
        return "Not found", 404
    resp = send_file(resolved, as_attachment=True, download_name=f"{name}.pdf")
    resp.headers["Cache-Control"] = "no-store"
    resp.headers.pop("ETag", None)
    resp.headers.pop("Last-Modified", None)
    return resp


@bp.route("/api/refresh-zines", methods=["POST"])
def api_refresh_zines():
    cfg               = load_cfg()
    back_descriptions = cfg.get("back_description", {})
    series_titles     = flatten_titles(cfg.get("series_titles", {}))

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

        last_img_idx = 1
        for i, item in enumerate(layout):
            if item.get("url"):
                last_img_idx = i
        M_count     = last_img_idx - 1
        base_len    = (last_img_idx + 5) if M_count % 2 == 0 else (last_img_idx + 4)
        desired_len = ((base_len + 3) // 4) * 4
        layout      = layout[:desired_len]
        while len(layout) < desired_len:
            ai = len(layout)
            layout.append({
                "url": None, "path": None,
                "x": (ai % 2) * HALF_W, "y": 0,
                "w": HALF_W, "h": canvas_h,
                "rot": 0, "brightness": 100, "contrast": 100, "white": True,
            })

        num_spreads = math.ceil(len(layout) / 2)
        while len(texts) < num_spreads:
            texts.append({"left":  {"text": "", "x": 0,      "y": canvas_h // 2 - 12},
                          "right": {"text": "", "x": HALF_W, "y": 12}})

        title_text = series_titles.get(series, {}).get(lang, "")
        if not title_text:
            for fl in ("pt", "en", "es", "fr"):
                title_text = series_titles.get(series, {}).get(fl, "")
                if title_text:
                    break
        bd_text = (back_descriptions.get(lang)
                   or back_descriptions.get("pt")
                   or back_descriptions.get("en") or "")

        if title_text:
            texts[0]["left"]["text"]  = title_text.strip().upper()
        if bd_text:
            texts[0]["right"]["text"] = bd_text.strip().upper()
        if texts[0]["left"].get("x") is None:
            texts[0]["left"]["x"] = 0
        if texts[0]["left"].get("y") is None:
            texts[0]["left"]["y"] = canvas_h // 2 - 12
        # BD_MARGIN=28 matches CANVAS_MARGIN in app.js and _make_zine_pdf
        texts[0]["right"]["x"] = HALF_W + 28
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

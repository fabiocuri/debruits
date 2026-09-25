#!/usr/bin/env python3
"""
make_template.py — generate DE BRUITS templates from Final/ composite images.

Layout, format sizes, and text are read from config.json (edited via the UI
or by hand). Change config.json once — every format updates automatically.

Usage
-----
    python3 make_template.py ALGAS   postcard pt          # front + back
    python3 make_template.py CONCHAS poster   en --side front
    python3 make_template.py PLANTAS fanzine  pt          # numbered pages
"""

import argparse
import json
import math
import random
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT      = Path(__file__).resolve().parent
FONT_PATH = ROOT / "DebruitsRegular-Handwritten.ttf"

# ── Config (read from config.json, with hardcoded defaults as fallback) ────────

_DEFAULT_LAYOUT = {
    "front_margin": 0.020, "front_gap": 0.010,
    "stamp_right": 0.020, "stamp_bottom": 0.012, "stamp_size": 0.048,
    "vline_x": 0.932, "vline_y0": 0.060, "vline_y1": 0.940,
    "hline_y": 0.908, "hline_x0": 0.050, "hline_x1": 0.330,
    "title_x": 0.050, "title_y": 0.916, "title_size": 0.018,
    "line_width": 2,
}
_DEFAULT_FORMATS = {
    "postcard":   {"size_px": (1240, 1949), "has_back": True},  # 10.5×16.5 cm @ 300 dpi
    "mini-poster": {"size_px": (1949, 3189), "has_back": True},  # 16.5×27 cm   @ 300 dpi
    "poster":     {"size_px": (2480, 3602), "has_back": True},  # 21×30.5 cm   @ 300 dpi
}
_DEFAULT_TITLES = {
    "ALGAS":    {"pt": "ALGAS DE PERNAMBUCO",    "en": "ALGAE FROM PERNAMBUCO"},
    "CONCHAS":  {"pt": "CONCHAS DE PERNAMBUCO",  "en": "SHELLS FROM PERNAMBUCO"},
    "FRUTAS":   {"pt": "FRUTAS DE PERNAMBUCO",   "en": "FRUITS FROM PERNAMBUCO"},
    "PLANTAS":  {"pt": "PLANTAS DE PERNAMBUCO",  "en": "PLANTS FROM PERNAMBUCO"},
    "ABACAXI":  {"pt": "ABACAXI DE PERNAMBUCO",  "en": "PINEAPPLE FROM PERNAMBUCO"},
    "PIMENTAO": {"pt": "PIMENTAO DE PERNAMBUCO", "en": "BELL PEPPER FROM PERNAMBUCO"},
    "OSTRA":    {"pt": "OSTRA DE PERNAMBUCO",    "en": "OYSTER FROM PERNAMBUCO"},
}


def _load_cfg():
    try:
        with open(ROOT / "config.json") as f:
            raw = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        raw = {}

    layout = {**_DEFAULT_LAYOUT, **raw.get("layout", {})}

    formats = _DEFAULT_FORMATS.copy()
    for k, v in raw.get("formats", {}).items():
        formats[k] = {**v, "size_px": tuple(v["size_px"])}

    raw_titles = {**_DEFAULT_TITLES, **raw.get("series_titles", {})}
    titles = {}
    for s, langs in raw_titles.items():
        titles[s] = {}
        for l, v in langs.items():
            titles[s][l] = v[0] if isinstance(v, list) else v

    data_root = Path(raw.get("data_root", ROOT.parent.parent))
    final_dir = data_root / "Final"
    tmpl_dir  = data_root / "Templates"
    return layout, formats, titles, final_dir, tmpl_dir


LAYOUT, FORMATS, SERIES_TITLES, FINAL_DIR, TMPL_DIR = _load_cfg()


_GEO_PREP = re.compile(
    r'\b(DE|DAS|DOS|FROM|VON|VAN|DI|DES|DEL|DEN|DEGLI|DU|DES)\b',
    re.IGNORECASE,
)


def _split_title_for_print(title: str) -> str:
    """Insert '\n' before the last geographic preposition when >1 word precedes it."""
    matches = list(_GEO_PREP.finditer(title))
    if matches:
        m = matches[-1]
        before = title[:m.start()].strip()
        if len(before.split()) > 1:
            return before + '\n' + title[m.start():].strip()
    return title


# ── Helpers ────────────────────────────────────────────────────────────────────

def collect_images(series: str) -> list:
    series_dir = FINAL_DIR / series.upper()
    if series_dir.is_dir():
        paths = sorted(series_dir.glob("*_upscaled.png"))
        if not paths:
            paths = sorted(series_dir.glob("*.png"))
    else:
        paths = sorted(FINAL_DIR.glob(f"{series.upper()}_*_upscaled.png"))
        if not paths:
            paths = sorted(FINAL_DIR.glob(f"{series.upper()}_*.png"))
    return paths


def load_font(size_px: int):
    try:
        return ImageFont.truetype(str(FONT_PATH), max(1, size_px))
    except OSError:
        return ImageFont.load_default()


def place_stamp(canvas: Image.Image, W: int, H: int,
                color=(0, 0, 0, 255), icon_path: str = None, icon_scale: float = 1.0) -> None:
    L         = LAYOUT
    base_size = round(L["stamp_size"] * W)
    x1        = W - round(L["stamp_right"]  * W)
    y1        = H - round(L["stamp_bottom"] * H)

    if icon_path:
        try:
            size    = round(base_size * icon_scale)
            x0, y0 = x1 - size, y1 - size
            icon    = Image.open(icon_path).convert("RGBA")
            icon    = icon.resize((size, size), Image.LANCZOS)
            canvas.alpha_composite(icon, (x0, y0))
            # Draw brand name above icon: "d b u t" / "e r i s"
            draw      = ImageDraw.Draw(canvas)
            font      = load_font(max(1, round(size * 0.28)))
            lines    = ["D B U T", "E R I S"]
            line_gap = max(2, round(size * 0.10))
            icon_gap = max(4, round(size * 0.15))
            bboxes   = [draw.textbbox((0, 0), ln, font=font) for ln in lines]
            ths      = [b[3] - b[1] for b in bboxes]
            # Anchor: visual bottom of last line = y0 - icon_gap (accounts for b[1] offset)
            ty = (y0 - icon_gap - bboxes[-1][3]
                  - sum(ths[:-1]) - line_gap * (len(lines) - 1))
            for line, bb in zip(lines, bboxes):
                tw  = bb[2] - bb[0]
                th  = bb[3] - bb[1]
                tx  = x0 + (size - tw) // 2
                draw.text((tx, ty), line, font=font, fill=color)
                ty += th + line_gap
            return
        except Exception:
            pass  # fall through to text stamp

    size    = base_size
    x0, y0 = x1 - size, y1 - size

    draw = ImageDraw.Draw(canvas)
    draw.rectangle([x0, y0, x1, y1], outline=color, width=2)
    font   = load_font(round(size * 0.26))
    lines  = ["DE", "BRUITS"]
    bboxes = [draw.textbbox((0, 0), ln, font=font) for ln in lines]
    gap    = 2
    total  = sum(b[3] - b[1] for b in bboxes) + gap * (len(lines) - 1)
    ty     = y0 + (size - total) // 2
    for line, bbox in zip(lines, bboxes):
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        draw.text((x0 + (size - tw) // 2, ty), line, font=font, fill=color)
        ty += th + gap


# ── Builders ───────────────────────────────────────────────────────────────────

def build_front(series: str, size_px: tuple, icon_path: str = None, icon_scale: float = 1.0) -> Image.Image:
    W, H = size_px
    L    = LAYOUT
    canvas = Image.new("RGBA", (W, H), (255, 255, 255, 255))
    imgs   = collect_images(series)

    if not imgs:
        print(f"  Warning: no images found for '{series}' in {FINAL_DIR}")
    else:
        n      = len(imgs)
        n_cols = math.ceil(math.sqrt(n))
        n_rows = math.ceil(n / n_cols)
        m      = round(L["front_margin"] * min(W, H))
        gap    = round(L["front_gap"]    * min(W, H))
        cell_w = (W - 2 * m - (n_cols - 1) * gap) // n_cols
        cell_h = (H - 2 * m - (n_rows - 1) * gap) // n_rows

        for idx, path in enumerate(imgs):
            row = idx // n_cols
            col = idx % n_cols
            cx  = m + col * (cell_w + gap)
            cy  = m + row * (cell_h + gap)

            img   = Image.open(path).convert("RGBA")
            scale = min(cell_w / img.width, cell_h / img.height)
            nw    = max(1, round(img.width  * scale))
            nh    = max(1, round(img.height * scale))
            img   = img.resize((nw, nh), Image.LANCZOS)
            canvas.alpha_composite(img, (cx + (cell_w - nw) // 2,
                                         cy + (cell_h - nh) // 2))

    place_stamp(canvas, W, H, icon_path=icon_path, icon_scale=icon_scale)
    return canvas


def build_front_scattered(series: str, size_px: tuple, count: int, icon_path: str = None, icon_scale: float = 1.0, forced_path: str = None):
    """Returns (image, layout) where layout is [{path, x, y, w, h}, ...]."""
    W, H   = size_px
    L      = LAYOUT
    canvas = Image.new("RGBA", (W, H), (255, 255, 255, 255))
    all_imgs = collect_images(series)

    # Pre-compute stamp bounding box so images never overlap it
    _base = round(L["stamp_size"] * W)
    _sz   = round(_base * icon_scale) if icon_path else _base
    _sx1  = W - round(L["stamp_right"]  * W)
    _sy1  = H - round(L["stamp_bottom"] * H)
    _sx0, _sy0 = _sx1 - _sz, _sy1 - _sz

    if not all_imgs:
        print(f"  Warning: no images found for '{series}' in {FINAL_DIR}")
        place_stamp(canvas, W, H, icon_path=icon_path, icon_scale=icon_scale)
        return canvas, []

    if forced_path:
        selected = [Path(forced_path)]
    else:
        selected = random.sample(all_imgs, min(count, len(all_imgs)))
        random.shuffle(selected)
    n      = len(selected)
    n_cols = math.ceil(math.sqrt(n))
    n_rows = math.ceil(n / n_cols)
    m      = round(L["front_margin"] * min(W, H))
    gap    = round(L["front_gap"]    * min(W, H))
    cell_w = (W - 2 * m - (n_cols - 1) * gap) // n_cols
    cell_h = (H - 2 * m - (n_rows - 1) * gap) // n_rows

    layout = []
    for idx, path in enumerate(selected):
        row = idx // n_cols
        col = idx % n_cols
        cx  = m + col * (cell_w + gap)
        cy  = m + row * (cell_h + gap)

        img_orig = Image.open(path).convert("RGBA")
        iw, ih   = img_orig.size
        scale    = min(cell_w / iw, cell_h / ih)
        nw       = max(1, round(iw * scale))
        nh       = max(1, round(ih * scale))
        ox       = (cell_w - nw) // 2
        oy       = (cell_h - nh) // 2
        px, py   = cx + ox, cy + oy

        # If this image would overlap the stamp zone, shift it away
        if px < _sx1 and px + nw > _sx0 and py < _sy1 and py + nh > _sy0:
            shift_l = px + nw - _sx0  # x-axis overlap amount
            shift_u = py + nh - _sy0  # y-axis overlap amount
            if shift_l <= shift_u:
                px = max(cx, px - shift_l)
            else:
                py = max(cy, py - shift_u)
            # Edge case: still overlapping — scale down to fit non-stamp area
            if px < _sx1 and px + nw > _sx0 and py < _sy1 and py + nh > _sy0:
                max_w = max(1, _sx0 - cx)
                max_h = max(1, _sy0 - cy)
                s2 = min(max_w / iw, max_h / ih)
                nw = max(1, round(iw * s2))
                nh = max(1, round(ih * s2))
                px = cx + (max_w - nw) // 2
                py = cy + (max_h - nh) // 2

        img = img_orig.resize((nw, nh), Image.LANCZOS)
        canvas.alpha_composite(img, (max(0, px), max(0, py)))
        layout.append({"path": str(path), "x": max(0, px), "y": max(0, py), "w": nw, "h": nh})

    place_stamp(canvas, W, H, icon_path=icon_path, icon_scale=icon_scale)
    return canvas, layout


def _adjust_image(img: Image.Image, brightness: float, contrast: float, bw: bool) -> Image.Image:
    from PIL import ImageEnhance
    if bw:
        r, g, b, a = img.split()
        gray = Image.merge("RGB", (r, g, b)).convert("L")
        img = Image.merge("RGBA", (gray, gray, gray, a))
    if brightness != 100:
        img = ImageEnhance.Brightness(img).enhance(brightness / 100.0)
    if contrast != 100:
        img = ImageEnhance.Contrast(img).enhance(contrast / 100.0)
    return img


def compose_layout(placements: list, size_px: tuple, icon_path: str = None, icon_scale: float = 1.0) -> Image.Image:
    """Re-composite a front image from dragged placements."""
    W, H   = size_px
    canvas = Image.new("RGBA", (W, H), (255, 255, 255, 255))
    for p in placements:
        img = Image.open(p["path"]).convert("RGBA")
        img = img.resize((int(p["w"]), int(p["h"])), Image.LANCZOS)
        img = _adjust_image(img,
                            float(p.get("brightness", 100)),
                            float(p.get("contrast",   100)),
                            bool(p.get("bw", False)))
        rot = float(p.get("rot", 0))
        if rot:
            img = img.rotate(-rot, expand=True, resample=Image.BICUBIC)
        # Center the (possibly rotated) image over the original bounding box center
        cx = int(p["x"]) + int(p["w"]) // 2
        cy = int(p["y"]) + int(p["h"]) // 2
        px = cx - img.width  // 2
        py = cy - img.height // 2
        # Clip to canvas bounds
        sx = max(0, -px); sy = max(0, -py)
        ex = min(img.width,  W - px)
        ey = min(img.height, H - py)
        if ex > sx and ey > sy:
            canvas.alpha_composite(img.crop((sx, sy, ex, ey)), (max(0, px), max(0, py)))
    place_stamp(canvas, W, H, icon_path=icon_path, icon_scale=icon_scale)
    return canvas


def _wrap_text(text: str, font, max_width: int, draw: ImageDraw.ImageDraw) -> list:
    """Word-wrap text to fit within max_width. Preserves paragraph breaks."""
    lines = []
    for paragraph in text.split("\n"):
        words = paragraph.split()
        if not words:
            lines.append("")
            continue
        current = ""
        for word in words:
            test = (current + " " + word).strip()
            w = draw.textbbox((0, 0), test, font=font)[2]
            if w <= max_width:
                current = test
            else:
                if current:
                    lines.append(current)
                current = word
        if current:
            lines.append(current)
    return lines


def build_back(series: str, lang: str, size_px: tuple, title: str = None, footer_text: str = None) -> Image.Image:
    W, H = size_px
    L    = LAYOUT
    canvas = Image.new("RGBA", (W, H), (255, 255, 255, 255))
    draw   = ImageDraw.Draw(canvas)
    lw     = int(L["line_width"])

    vx = round(L["vline_x"] * W)
    draw.line([(vx, round(L["vline_y0"] * H)), (vx, round(L["vline_y1"] * H))],
              fill=(0, 0, 0, 255), width=lw)

    hy = round(L["hline_y"] * H)
    draw.line([(round(L["hline_x0"] * W), hy), (round(L["hline_x1"] * W), hy)],
              fill=(0, 0, 0, 255), width=lw)

    if title is None:
        title = SERIES_TITLES.get(series.upper(), {}).get(lang, series.upper())
    title = _split_title_for_print(title)
    title_font = load_font(max(10, round(L["title_size"] * H)))
    _tdraw = draw.multiline_text if '\n' in title else draw.text
    _tdraw((round(L["title_x"] * W), round(L["title_y"] * H)),
           title, font=title_font, fill=(0, 0, 0, 255),
           stroke_width=2, stroke_fill=(0, 0, 0, 255))

    if footer_text:
        footer_font_size = max(10, round(L["title_size"] * H * 0.85))
        footer_font = load_font(footer_font_size)
        margin_x    = round(0.05 * W)
        max_w       = vx - 2 * margin_x
        line_h      = round(footer_font_size * 1.4)
        para_gap    = round(footer_font_size * 0.6)

        wrapped = _wrap_text(footer_text, footer_font, max_w, draw)

        # Measure total height
        total_h = 0
        for ln in wrapped:
            total_h += para_gap if ln == "" else line_h

        # Anchor text block to bottom of content area; clamp so it never goes above top margin
        top_margin = round(L["vline_y0"] * H) + round(0.02 * H)
        bottom_y   = hy - round(0.01 * H)
        ty         = max(top_margin, bottom_y - total_h)

        for ln in wrapped:
            if ln == "":
                ty += para_gap
                continue
            draw.text((margin_x, ty), ln, font=footer_font, fill=(0, 0, 0, 255))
            ty += line_h

    return canvas




# ── CLI ────────────────────────────────────────────────────────────────────────

def _save(img: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img.convert("RGB").save(path, "PNG", dpi=(600, 600), compress_level=1)
    try:
        label = path.relative_to(ROOT)
    except ValueError:
        label = path
    print(f"  → {label}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate DE BRUITS templates.")
    ap.add_argument("series",  help="Series name, e.g. ALGAS")
    ap.add_argument("format",  choices=list(FORMATS))
    ap.add_argument("lang",    choices=["pt", "en"])
    ap.add_argument("--side",  choices=["front", "back"], default="front")
    ap.add_argument("--count", type=int, default=None, help="Number of images to place (random scatter)")
    ap.add_argument("--out",   help="Override output directory")
    args = ap.parse_args()

    series  = args.series.upper()
    fmt_cfg = FORMATS[args.format]
    size_px = fmt_cfg["size_px"]
    lang    = args.lang
    lang_dir = "PORTUGUES" if lang == "pt" else "ENGLISH"

    print(f"\nDE BRUITS — {args.format.upper()} / {lang.upper()} / {series}")
    print("-" * 50)

    out_dir = Path(args.out) if args.out else TMPL_DIR / args.format.upper() / lang_dir
    for side in [args.side]:
        if side == "front":
            img = (build_front_scattered(series, size_px, args.count)[0]
                   if args.count is not None
                   else build_front(series, size_px))
        else:
            img = build_back(series, lang, size_px)
        _save(img, out_dir / f"{series}_{side.upper()}.png")

    print("\nDone.")


if __name__ == "__main__":
    main()

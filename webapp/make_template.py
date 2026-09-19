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
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT      = Path(__file__).resolve().parent
FONT_PATH = ROOT / "Debruits-Regular.ttf"

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
    "postcard": {"size_px": (1754, 2480), "has_back": True},
    "poster":   {"size_px": (1754, 2480), "has_back": True},
    "fanzine":  {"size_px": (1754, 2480), "has_back": False},
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

    titles = {**_DEFAULT_TITLES, **raw.get("series_titles", {})}

    data_root = Path(raw.get("data_root", ROOT.parent.parent))
    final_dir = data_root / "Final"
    tmpl_dir  = data_root / "Templates"
    return layout, formats, titles, final_dir, tmpl_dir


LAYOUT, FORMATS, SERIES_TITLES, FINAL_DIR, TMPL_DIR = _load_cfg()


# ── Helpers ────────────────────────────────────────────────────────────────────

def collect_images(series: str) -> list:
    paths = sorted(FINAL_DIR.glob(f"{series.upper()}_*_upscaled.png"))
    if not paths:
        paths = sorted(FINAL_DIR.glob(f"{series.upper()}_*.png"))
    return paths


def load_font(size_px: int):
    try:
        return ImageFont.truetype(str(FONT_PATH), max(1, size_px))
    except OSError:
        return ImageFont.load_default()


def place_stamp(draw, W: int, H: int) -> None:
    L    = LAYOUT
    size = round(L["stamp_size"] * W)
    x1   = W - round(L["stamp_right"]  * W)
    y1   = H - round(L["stamp_bottom"] * H)
    x0, y0 = x1 - size, y1 - size

    draw.rectangle([x0, y0, x1, y1], outline=(255, 255, 255, 255), width=2)

    font   = load_font(round(size * 0.26))
    lines  = ["DE", "BRUITS"]
    bboxes = [draw.textbbox((0, 0), ln, font=font) for ln in lines]
    gap    = 2
    total  = sum(b[3] - b[1] for b in bboxes) + gap * (len(lines) - 1)
    ty     = y0 + (size - total) // 2
    for line, bbox in zip(lines, bboxes):
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        draw.text((x0 + (size - tw) // 2, ty), line,
                  font=font, fill=(255, 255, 255, 255))
        ty += th + gap


# ── Builders ───────────────────────────────────────────────────────────────────

def build_front(series: str, size_px: tuple) -> Image.Image:
    W, H = size_px
    L    = LAYOUT
    canvas = Image.new("RGBA", (W, H), (0, 0, 0, 255))
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

    place_stamp(ImageDraw.Draw(canvas), W, H)
    return canvas


def build_back(series: str, lang: str, size_px: tuple) -> Image.Image:
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

    title = SERIES_TITLES.get(series.upper(), {}).get(lang, series.upper())
    font  = load_font(max(10, round(L["title_size"] * H)))
    draw.text((round(L["title_x"] * W), round(L["title_y"] * H)),
              title, font=font, fill=(0, 0, 0, 255))

    return canvas


def build_fanzine_pages(series: str, size_px: tuple) -> list:
    W, H   = size_px
    L      = LAYOUT
    m      = round(L["front_margin"] * min(W, H))
    pages  = []

    for path in collect_images(series):
        canvas = Image.new("RGBA", (W, H), (0, 0, 0, 255))
        img    = Image.open(path).convert("RGBA")
        scale  = min((W - 2 * m) / img.width, (H - 2 * m) / img.height)
        nw     = max(1, round(img.width  * scale))
        nh     = max(1, round(img.height * scale))
        img    = img.resize((nw, nh), Image.LANCZOS)
        canvas.alpha_composite(img, ((W - nw) // 2, (H - nh) // 2))
        place_stamp(ImageDraw.Draw(canvas), W, H)
        pages.append(canvas)

    return pages


# ── CLI ────────────────────────────────────────────────────────────────────────

def _save(img: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, "PNG")
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
    ap.add_argument("--side",  choices=["front", "back", "both"], default="both")
    ap.add_argument("--out",   help="Override output directory")
    args = ap.parse_args()

    series  = args.series.upper()
    fmt_cfg = FORMATS[args.format]
    size_px = fmt_cfg["size_px"]
    lang    = args.lang
    lang_dir = "PORTUGUES" if lang == "pt" else "ENGLISH"

    print(f"\nDE BRUITS — {args.format.upper()} / {lang.upper()} / {series}")
    print("-" * 50)

    if args.format == "fanzine":
        out_dir = Path(args.out) if args.out else TMPL_DIR / "FANZINE" / series
        pages   = build_fanzine_pages(series, size_px)
        if not pages:
            print(f"  No images found for '{series}'.")
            return
        for i, pg in enumerate(pages, 1):
            _save(pg, out_dir / f"{i}.png")
        zine = TMPL_DIR / "FANZINE" / "make_zine.py"
        try:
            out_label = out_dir.relative_to(ROOT)
        except ValueError:
            out_label = out_dir
        print(f"\n  {len(pages)} pages written. To assemble:")
        print(f"  python3 {zine.relative_to(ROOT)} {out_label}")
    else:
        out_dir = Path(args.out) if args.out else TMPL_DIR / args.format.upper() / lang_dir
        sides   = ["front", "back"] if args.side == "both" else [args.side]
        if not fmt_cfg["has_back"]:
            sides = [s for s in sides if s != "back"]
        for side in sides:
            img  = (build_front(series, size_px) if side == "front"
                    else build_back(series, lang, size_px))
            _save(img, out_dir / f"{series}_{side.upper()}.png")

    print("\nDone.")


if __name__ == "__main__":
    main()

# DE BRUITS — webapp

Local dashboard for the DE BRUITS project.

## Running with Docker (recommended)

From the repo root:

```bash
docker compose up --build -d
```

Open **http://localhost:5000**. Logs:

```bash
docker compose logs -f
```

Stop:

```bash
docker compose down
```

## Running without Docker

Python 3.10+. Install dependencies:

```bash
pip install flask pillow img2pdf pypdf gunicorn
```

Then from `webapp/`:

```bash
python3 app.py
```

Open **http://localhost:5000**

---

## Configuration

### `config.json`

| Key | Description |
|-----|-------------|
| `data_root` | Absolute path to the DE BRUITS working directory (where `Templates/`, `Final/`, and `Zines/` live) |
| `layout` | Template layout parameters (margins, stamp position, etc.) |
| `formats` | Canvas sizes per format |
| `series_titles` | Display titles per series per language |
| `back_footer` | Per-language text for the back of prints |
| `back_description` | Per-language description text for the back of zines |

---

## Tabs

### Home (theme grid)
Landing view showing all image themes as tiles. Click a tile to browse its photos and set the cover thumbnail.

### Prints
Browse and manage generated print templates. Toolbar actions:

- **+ Create Print** — pick format / theme / language / icon, preview and save as PDF
- **Download All** — zip all saved PDFs
- **Theme Titles** — edit the per-language titles used on print backs
- **Back Description** — edit the back footer text
- **Refresh** — re-render all existing prints from their saved layouts
- **Select All / Delete Selected**

Filter sidebar: Theme, Format (Postcard / Mini Poster / Poster), Language, Icon.

### Zines
Browse and manage generated zines. Same toolbar actions as Prints (adapted for zines).

**+ Create Zine** opens a two-step modal:

**Step 1 — configure:**
- **Format**: Mini Poster (A5 pages, A4 landscape spread) or Poster (A4 pages, A3 landscape spread)
- **Theme**: pick a single theme, or **MISC** to mix all themes
  - **MISC → N**: ALL (every photo from each theme) or N (pick N photos per theme)
  - **MISC → M**: number of photos per half-page (M > 1 stacks images vertically in each slot)
- **Language**: PT / EN / ES / FR
- **Images**: check/uncheck individual photos, set brightness/contrast, mark one as cover
- **Icon**: optional stamp on the front cover

**Step 2 — preview:**
- Navigate spreads with `<` / `>`
- Drag photos and text labels to reposition
- Scale (global) and Cover scale sliders
- B&W toggle
- Save → writes PDF + sidecar JSON to `Zines/`

### Paper
GSM reference tables for each format (Postcard / Mini Poster / Poster).

---

## Directory layout

```
webapp/
├── app.py                     Flask server
├── make_template.py           Template generator (Pillow)
├── config.json                Layout, formats, series titles, paths
├── Debruits-Regular.ttf       Custom typeface (layout elements)
├── DebruitsRegular-Handwritten.ttf  Custom typeface (text labels)
├── Debruits-Extended.ttf      Custom typeface (extended)
├── thumbnails.json            Per-theme cover image overrides
├── static/
│   ├── index.html
│   ├── style.css
│   └── app.js
└── interests/                 (unused legacy directory)
```

---

## Formats

| Format | Page size | Spread (PDF) | Canvas px |
|--------|-----------|--------------|-----------|
| Postcard | 10.5 × 16.5 cm | — | 1240 × 1949 |
| Mini Poster | 16.5 × 27 cm | A4 landscape | 680 × 382 (preview) |
| Poster | 21 × 30.5 cm | A3 landscape | 680 × 484 (preview) |

Zine spreads are rendered at 300 dpi and saved as booklet-imposed PDFs (print → fold).

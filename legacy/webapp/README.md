# DE BRUITS — webapp

Local dashboard for the DE BRUITS project. See [ARCHITECTURE.md](../ARCHITECTURE.md) for the full system diagram.

## Running

The primary way to run the app is via Kubernetes — see the [root README](../README.md).

For local development without k8s:

```bash
cd webapp
DATA_ROOT=../data python3 app.py
```

Open **http://localhost:5000**

---

## Configuration

### `config.json`

| Key | Description |
|-----|-------------|
| `layout` | Template layout parameters (margins, stamp position, line widths) |
| `formats` | Canvas sizes per format (overrides hardcoded defaults) |
| `series_titles` | Display titles per series per language |
| `back_footer` | Per-language text for the back of prints |
| `back_description` | Per-language description text for the back of zines |

The data directory is controlled by the `DATA_ROOT` environment variable (not `config.json`). The k8s deployment sets it to `/data`; the local fallback is `../data` relative to `webapp/`.

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
├── app.py                     Flask server + all API routes
├── make_template.py           Pillow compositor (layout, back, stamp)
├── config.json                Layout params, format sizes, series titles
├── DebruitsRegular-Handwritten.ttf  Custom typeface
├── DebruitsRegular-Handwritten.sfd  FontForge source for the typeface
├── thumbnails.json            Per-theme cover image overrides
├── build_font.sh              Rebuild TTF from SFD
├── static/
│   ├── index.html
│   ├── style.css
│   └── app.js
└── interests/                 Optional standalone FastAPI dashboard (port 5001)
    ├── app/                   Claude API, Gmail, Calendar, GitHub, Spotify clients
    └── templates/index.html
```

---

## Formats

| Format | Page size | Canvas px (300 dpi) | Paper fit |
|--------|-----------|---------------------|-----------|
| Postcard | 10.5 x 16.5 cm | 2480 x 3898 | 10 x 15 cm standard |
| Mini Poster | 16.5 x 27 cm | 3898 x 6378 | A5 - 14.8 x 21 cm |
| Poster | 21 x 30.5 cm | 4961 x 7205 | A4 - 21 x 29.7 cm |

Zine spreads are imposed into booklet-order PDFs (print → fold in half).

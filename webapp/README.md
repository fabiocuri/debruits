# DE BRUITS — webapp

Local dashboard for the DE BRUITS project. One command starts everything:

```bash
python3 app.py
```

- **http://localhost:5000** — DE BRUITS dashboard (templates, images, parameters)
- **http://localhost:5001** — Interests panel (embedded in the Info tab)

---

## Prerequisites

Python 3.10+. Install all dependencies in one shot:

```bash
pip install flask pillow fastapi "uvicorn[standard]" anthropic jinja2 \
            python-dotenv icalendar recurring-ical-events markupsafe
```

---

## Configuration

### `config.json`

| Key | Description |
|-----|-------------|
| `data_root` | Absolute path to the DE BRUITS working directory (where `Templates/` and `Final/` live) |
| `info_dashboard_url` | URL for the interests dashboard subprocess (default `http://localhost:5001`) |
| `layout` | Template layout parameters (editable live from the Parameters tab) |
| `formats` | Canvas sizes per format (postcard / poster / fanzine) |
| `series_titles` | Display titles per series in PT and EN |

### `interests/.env`

Copy `.env.example` and fill in your credentials:

```bash
cp interests/.env.example interests/.env
```

Required for the Info tab to show live data:

| Variable | Purpose |
|----------|---------|
| `ANTHROPIC_API_KEY` | Claude API — powers the daily digest tasks |
| `GMAIL_ADDRESS` / `GMAIL_APP_PASSWORD` | Read-only inbox panel |
| `CALENDAR_ICS_URL` | Google Calendar secret ICS feed |
| `GITHUB_TOKEN` | GitHub review requests / notifications panel |
| `SPOTIFY_CLIENT_ID` / `SPOTIFY_CLIENT_SECRET` | Globe music player |

See `interests/.env.example` for the full list and optional overrides.

---

## Generating templates

Via the dashboard: click **+ Generate**, pick series / format / language, click Generate.

Via CLI:

```bash
python3 make_template.py ALGAS postcard pt          # front + back
python3 make_template.py CONCHAS poster en --side front
python3 make_template.py PLANTAS fanzine pt         # numbered pages
```

Templates are saved to `Templates/` under the configured `data_root`.

---

## Directory layout

```
webapp/
├── app.py                 Flask server + interests subprocess launcher
├── make_template.py       Template generator (Pillow)
├── config.json            Layout, formats, series titles, paths
├── Debruits-Regular.ttf   Custom typeface
├── static/
│   ├── index.html
│   ├── style.css
│   └── app.js
└── interests/             Personal interests dashboard (FastAPI / uvicorn)
    ├── app/               FastAPI application package
    ├── templates/         Jinja2 HTML templates
    ├── .env               API credentials (not committed)
    └── .env.example       Credential reference
```

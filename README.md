# DE BRUITS

DE BRUITS is an art project combining microscopy and macro photography to produce postcards, posters, and fanzines.

**Author:** Fabio Curi · fcuri91@gmail.com

---

## Quick start

```bash
cd webapp
pip install flask pillow pypdf img2pdf
python3 app.py
```

Open **http://localhost:5000**

---

## Repository structure

```
debruits/
├── webapp/       Main dashboard — template generation, image browser, zine editor
└── legacy/       Archived experiments — not maintained
```

---

## Configuration

Edit `webapp/config.json`:

| Key | Description |
|-----|-------------|
| `data_root` | Absolute path to your working directory (where `Final/` lives) |
| `info_dashboard_url` | Interests dashboard URL (default `http://localhost:5001`) |

---

## CLI template generation

```bash
cd webapp
python3 make_template.py ALGAS postcard pt
python3 make_template.py CONCHAS poster en --side front
```

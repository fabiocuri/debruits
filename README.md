# DE BRUITS

Art project combining microscopy and macro photography to produce postcards, posters, and zines.  
**Author:** Fabio Curi · fcuri91@gmail.com

---

## Contents

- [Structure](#structure)
- [Run — Kubernetes](#run--kubernetes)
- [Run — local](#run--local)
- [Configuration](#configuration)
- [Data layout](#data-layout)
- [API](#api)

---

## Structure

```
debruits/
│
├── src/
│   ├── api/
│   │   ├── __init__.py          App factory (create_app)
│   │   ├── paths.py             Centralised path constants
│   │   ├── utils.py             Shared utilities
│   │   ├── compositor.py        Pillow compositor — layout, back face, stamp
│   │   └── routes/
│   │       ├── assets.py        File serving
│   │       ├── config.py        /api/config
│   │       ├── images.py        /api/images, /api/covers, /api/thumbnails
│   │       ├── prints.py        /api/templates, /api/generate, /api/compose …
│   │       └── zines.py         /api/zines, /api/zine/create … + PDF imposition
│   ├── static/                  index.html · app.js · style.css
│   ├── fonts/                   DebruitsRegular-Handwritten.ttf
│   ├── config.json              Default config (seed — copied to data/ on first run)
│   └── thumbnails.json          Series thumbnails (runtime, gitignored)
│
├── infra/
│   ├── Dockerfile
│   ├── docker-compose.yml
│   ├── Caddyfile
│   └── k8s/
│       ├── deployment.yaml      hostPath /data, imagePullPolicy Never
│       └── service.yaml         NodePort 30500
│
├── docs/
├── data/                        Mounted at /data — gitignored
│   ├── Series/                  Source images  e.g. Series/ALGAS/
│   ├── Cover/                   Cover photos   e.g. Cover/ALGAS/
│   ├── Templates/               Saved print PDFs + sidecars
│   ├── Zines/                   Saved zine PDFs + sidecars
│   ├── icons/                   Stamp icon PNGs (PNG with transparency)
│   └── config.json              Live config — auto-seeded from src/config.json
│
├── legacy/                      Archived GAN experiments, Vue prototype, old webapp
├── pyproject.toml
└── uv.lock
```

---

## Run — Kubernetes

**First run**

```bash
minikube start
minikube mount "$(pwd)/data:/data" &

eval $(minikube docker-env)
docker build -f infra/Dockerfile -t debruits-app:latest .
kubectl apply -f infra/k8s/

minikube service debruits
```

**After a restart**

```bash
minikube start
minikube mount "$(pwd)/data:/data" &
kubectl apply -f infra/k8s/
minikube service debruits
```

**After code changes**

```bash
eval $(minikube docker-env)
docker build -f infra/Dockerfile -t debruits-app:latest .
kubectl rollout restart deployment/debruits
```

> `minikube mount` is required because the Docker driver runs inside a container — `hostPath` volumes are only reachable via the 9p mount.

---

## Run — local

```bash
uv sync
cd src
DATA_ROOT=../data uv run python -m api
```

Open **http://localhost:5000**

**CLI compositor**

```bash
DATA_ROOT=../data uv run python -m api.compositor ALGAS postcard pt
DATA_ROOT=../data uv run python -m api.compositor CONCHAS poster en --side back
```

---

## Configuration

The live config is `data/config.json`. On first run the app seeds it from `src/config.json` (the default shipped in the image) if it doesn't exist yet.

| Key | Description |
|-----|-------------|
| `series_titles` | Per-series titles per language |
| `back_footer` | Postcard back footer text per language |
| `back_description` | Zine back description per language |
| `layout` | Stamp position, margins, line widths |
| `formats` | Canvas pixel dimensions per format |

`DATA_ROOT` env var sets the data directory. k8s sets it to `/data`; local default is `../data` relative to `src/`.

---

## Data layout

```
data/
├── config.json                          ← live config (seeded from src/config.json)
├── icons/                               ← stamp icon PNGs (PNG with transparency)
├── Series/ALGAS/ALGAS_01_upscaled.png   ← source images per series
├── Cover/ALGAS/cover.jpg                ← cover photos per series
├── Templates/POSTCARD/PORTUGUES/        ← saved prints (auto-created)
│   ├── ALGAS.pdf
│   ├── ALGAS.json                       ← sidecar: layout, icon, lang …
│   └── ALGAS_thumb.png
└── Zines/ALGAS_20250101_120000.pdf      ← saved zines (auto-created)
```

---

## API

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Single-page app |
| GET | `/img/<path>` | Serve image from DATA_ROOT |
| GET | `/font/<name>` | Custom TTF font |
| GET | `/icon/<name>` | Stamp icon PNG |
| GET/PUT | `/api/config` | Read / update config.json |
| GET | `/api/images` | List all Series/ source images |
| GET | `/api/covers/<series>` | List Cover/<series>/ photos |
| GET | `/api/icons` | List stamp icons |
| GET/PUT | `/api/thumbnails` | Per-series thumbnail overrides |
| GET | `/api/templates` | List saved prints |
| DELETE | `/api/templates` | Delete prints |
| GET | `/api/layout-data/<path>` | Sidecar JSON for a saved print |
| POST | `/api/generate` | Preview layout (no disk write) |
| POST | `/api/compose` | Render + save print PDF |
| POST | `/api/refresh-prints` | Re-render all prints from sidecars |
| GET | `/api/download/<path>` | Download one print PDF |
| GET | `/api/download-all` | Zip all print PDFs |
| GET | `/api/zines` | List saved zines |
| POST | `/api/zine/create` | Render + save zine PDF |
| DELETE | `/api/zines/delete` | Delete zines |
| POST | `/api/refresh-zines` | Re-render all zines from sidecars |
| GET | `/api/zines/download/<name>` | Download one zine PDF |
| GET | `/api/zines/download-all` | Zip all zine PDFs |

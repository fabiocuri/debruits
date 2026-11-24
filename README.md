# DE BRUITS

Art project combining microscopy and macro photography to produce postcards, posters, and zines.

**Author:** Fabio Curi · fcuri91@gmail.com · [architecture →](docs/architecture.md)

---

## Repository structure

```
debruits/
│
├── src/                            Flask application
│   ├── api/
│   │   ├── __init__.py             App factory (create_app)
│   │   ├── paths.py                Centralised path constants
│   │   ├── utils.py                Shared utilities
│   │   ├── compositor.py           Pillow compositor — layout, back face, stamp
│   │   └── routes/
│   │       ├── assets.py           File serving — images, fonts, icons, downloads
│   │       ├── config.py           /api/config
│   │       ├── images.py           /api/images, /api/covers, /api/thumbnails
│   │       ├── prints.py           /api/templates, /api/generate, /api/compose …
│   │       └── zines.py            /api/zines, /api/zine/create … + PDF imposition
│   ├── static/
│   │   ├── index.html
│   │   ├── app.js
│   │   └── style.css
│   ├── fonts/
│   │   └── DebruitsRegular-Handwritten.ttf
│   ├── config.json                 Layout, formats, series titles
│   └── thumbnails.json             Series thumbnails — runtime, gitignored
│
├── infra/
│   ├── Dockerfile                  Builds image from src/ + icons/
│   ├── docker-compose.yml          Compose alternative — mounts ./data as /data
│   ├── Caddyfile                   Reverse-proxy config
│   └── k8s/
│       ├── deployment.yaml         hostPath /data, imagePullPolicy Never
│       └── service.yaml            NodePort 30500
│
├── docs/
│   └── architecture.md             System diagram and full API reference
│
├── data/                           Working data — mounted at /data, gitignored
│   ├── Series/                     Source images  e.g. Series/ALGAS/
│   ├── Cover/                      Cover photos   e.g. Cover/ALGAS/
│   ├── Templates/                  Saved print PDFs + JSON sidecars
│   └── Zines/                      Saved zine PDFs + JSON sidecars
│
├── icons/                          Stamp icon PNGs — baked into image, gitignored
├── legacy/                         Archived GAN experiments, Vue prototype, old webapp
├── pyproject.toml                  Dependencies (uv)
└── uv.lock
```

---

## Running with Kubernetes

### First run

```bash
# 1. Start minikube with the data directory mounted
minikube start
minikube mount "$(pwd)/data:/data" &

# 2. Build the image inside minikube's Docker context
eval $(minikube docker-env)
docker build -f infra/Dockerfile -t debruits-app:latest .

# 3. Deploy
kubectl apply -f infra/k8s/

# 4. Open in browser
minikube service debruits
```

### Subsequent runs (after a machine restart)

```bash
minikube start
minikube mount "$(pwd)/data:/data" &
kubectl apply -f infra/k8s/
minikube service debruits
```

### Rebuild after code changes

```bash
eval $(minikube docker-env)
docker build -f infra/Dockerfile -t debruits-app:latest .
kubectl rollout restart deployment/debruits
```

---

## Running locally

```bash
uv sync
cd src
DATA_ROOT=../data uv run python -m api
```

Open **http://localhost:5000**

---

## Configuration

Edit `src/config.json`:

| Key | Description |
|-----|-------------|
| `series_titles` | Per-series titles per language |
| `back_footer` | Postcard back footer text per language |
| `back_description` | Zine back description per language |
| `layout` | Stamp position, margins, line widths |
| `formats` | Canvas pixel dimensions per format |

`DATA_ROOT` env var controls where the app reads/writes data. The k8s deployment sets it to `/data`; local fallback is `../data` relative to `src/`.

---

## Data layout

```
data/Series/ALGAS/ALGAS_01_upscaled.png
data/Series/CONCHAS/CONCHAS_01_upscaled.png
data/Cover/ALGAS/cover.jpg
```

Stamp icons (PNG with transparency) go under `icons/`.

---

## CLI compositor

```bash
cd src
DATA_ROOT=../data uv run python -m api.compositor ALGAS postcard pt
DATA_ROOT=../data uv run python -m api.compositor CONCHAS poster en --side back
```

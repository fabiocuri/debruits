# DE BRUITS

DE BRUITS is an art project combining microscopy and macro photography to produce postcards, posters, and fanzines.

**Author:** Fabio Curi · fcuri91@gmail.com

See [docs/architecture.md](docs/architecture.md) for the full system diagram and API reference.

---

## Running with Kubernetes (primary)

### First run

```bash
# 1. Start minikube and mount the data directory
minikube start --mount --mount-string="$(pwd)/data:/data"

# 2. Mount needs a background process (keep this terminal open, or run with & )
minikube mount "$(pwd)/data:/data"

# 3. Build the image inside minikube's Docker context
eval $(minikube docker-env)
docker build -f infra/Dockerfile -t debruits-app:latest .

# 4. Deploy
kubectl apply -f infra/k8s/

# 5. Open in browser
minikube service debruits
```

The app will be available at the URL printed by `minikube service debruits` (NodePort 30500).

### Subsequent runs (after a machine restart)

```bash
# Start minikube — data mount must come back up too
minikube start
minikube mount "$(pwd)/data:/data" &

# If the image is still cached, just reapply
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

## Running locally (development fallback)

```bash
uv sync
cd src
DATA_ROOT=../data uv run python -m api
```

Open **http://localhost:5000**

---

## Repository structure

```
debruits/
├── src/                  Flask application source
│   ├── api/              Python package (Flask app factory + all routes)
│   │   ├── __init__.py   create_app() factory, index route
│   │   ├── paths.py      Centralised path constants
│   │   ├── utils.py      Shared utilities
│   │   ├── compositor.py Pillow compositor (layout, back face, stamp)
│   │   └── routes/       Blueprint modules (assets, config, images, prints, zines)
│   ├── static/           Frontend (index.html, app.js, style.css)
│   ├── fonts/            DebruitsRegular-Handwritten.ttf + source
│   ├── config.json       Layout params, format sizes, series titles
│   └── thumbnails.json   Series overview thumbnails (runtime-generated, gitignored)
├── infra/                Infrastructure files
│   ├── Dockerfile        Builds app image (copies src/, icons/, config.json)
│   ├── docker-compose.yml Compose alternative — mounts ./data as /data
│   ├── Caddyfile         Reverse-proxy config
│   └── k8s/
│       ├── deployment.yaml Deployment — hostPath /data, local image
│       └── service.yaml    NodePort service on port 30500
├── docs/
│   └── architecture.md   Full system diagram and API reference
├── data/                 Working data — mounted into k8s at /data (gitignored)
│   ├── Series/           Source images by series (ALGAS/, CONCHAS/, …)
│   ├── Cover/            Cover photos by series
│   ├── Templates/        Saved print PDFs + sidecars (auto-created by app)
│   └── Zines/            Saved zine PDFs + sidecars (auto-created by app)
├── icons/                Stamp icon PNGs (baked into Docker image, gitignored)
├── legacy/               Archived GAN experiments and Vue.js prototype (not in use)
├── pyproject.toml        Project metadata and dependencies (uv)
└── uv.lock               Locked dependency versions
```

---

## Configuration

Edit `src/config.json`:

| Key | Description |
|-----|-------------|
| `series_titles` | Per-series titles in each language |
| `back_footer` | Postcard back footer text per language |
| `back_description` | Zine back description per language |
| `layout` | Fine-tune stamp position, margins, line widths |
| `formats` | Override default pixel dimensions for postcard / mini-poster / poster |

The data directory is set via the `DATA_ROOT` environment variable — the k8s deployment sets it to `/data`, the local fallback is `../data` relative to `webapp/`.

---

## Image organisation

Place processed images under `data/Series/<SERIES>/`, e.g.:

```
data/Series/ALGAS/ALGAS_01_upscaled.png
data/Series/CONCHAS/CONCHAS_01_upscaled.png
```

Cover photos go under `data/Cover/<SERIES>/`.
Stamp icons (PNG with transparency) go under `icons/`.

---

## CLI template generation

```bash
cd src
DATA_ROOT=../data uv run python -m api.compositor ALGAS postcard pt
DATA_ROOT=../data uv run python -m api.compositor CONCHAS poster en --side front
```

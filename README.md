# DE BRUITS

DE BRUITS is an art project combining microscopy and macro photography to produce postcards, posters, and fanzines.

**Author:** Fabio Curi · fcuri91@gmail.com

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full system diagram and API reference.

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
docker build -t debruits-app:latest .

# 4. Deploy
kubectl apply -f k8s/

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
kubectl apply -f k8s/

minikube service debruits
```

### Rebuild after code changes

```bash
eval $(minikube docker-env)
docker build -t debruits-app:latest .
kubectl rollout restart deployment/debruits
```

---

## Running locally (development fallback)

```bash
cd webapp
pip install -r ../requirements.txt
DATA_ROOT=../data python3 app.py
```

Open **http://localhost:5000**

---

## Repository structure

```
debruits/
├── webapp/               Flask app — template generation, image browser, zine editor
│   ├── app.py            Server + all API routes
│   ├── make_template.py  Pillow compositor (layout, back face, stamp)
│   ├── config.json       Layout params, format sizes, series titles
│   ├── static/           Frontend (index.html, app.js, style.css)
│   ├── DebruitsRegular-Handwritten.ttf  Custom font (baked into Docker image)
│   └── interests/        Optional standalone FastAPI dashboard (port 5001)
├── data/                 Working data — mounted into k8s at /data (gitignored)
│   ├── Series/           Source images by series (ALGAS/, CONCHAS/, …)
│   ├── Cover/            Cover photos by series
│   ├── Templates/        Saved print PDFs + sidecars (auto-created by app)
│   └── Zines/            Saved zine PDFs + sidecars (auto-created by app)
├── icons/                Stamp icon PNGs (baked into Docker image, gitignored)
├── k8s/
│   ├── deployment.yaml   Deployment — hostPath /data, local image
│   └── service.yaml      NodePort service on port 30500
├── legacy/               Archived GAN experiments and Vue.js prototype (not in use)
├── ARCHITECTURE.md       Full system diagram and API reference
├── Dockerfile            Builds app image (copies webapp/ and icons/, not data/)
├── docker-compose.yml    Compose alternative — mounts ./data as /data
└── requirements.txt      Python dependencies
```

---

## Configuration

Edit `webapp/config.json`:

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
cd webapp
python3 make_template.py ALGAS postcard pt
python3 make_template.py CONCHAS poster en --side front
```

# Structure

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
├── legacy/                      Archived ML experiments (CAN, GAN)
├── pyproject.toml
└── uv.lock
```

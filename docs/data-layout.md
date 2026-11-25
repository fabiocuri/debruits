# Data layout

```
data/
├── config.json                          ← live config (seeded from src/config.json)
├── Icons/
│   ├── Icon_1.png                       ← stamp icon PNGs (PNG with transparency)
│   └── Icon_2.png
├── Series/
│   ├── ALGAS/ALGAS_1.png                ← source images per series
│   └── CONCHAS/CONCHAS_1.png
├── Cover/
│   ├── ALGAS/ALGAS.png                  ← cover photo per series
│   └── CONCHAS/CONCHAS.png
├── Templates/
│   └── POSTCARD/
│       ├── PORTUGUES/
│       │   ├── ALGAS_icon.pdf           ← saved print ({SERIES}_{ICON}.pdf)
│       │   ├── ALGAS_icon.json          ← sidecar: layout, icon, lang …
│       │   └── ALGAS_icon_thumb.png
│       └── FRANCAIS/
│           └── PLANTAS_icon.pdf
└── Zines/
    └── FRUTAS_YYYYMMDD_HHMMSS.pdf       ← saved zine PDF
```

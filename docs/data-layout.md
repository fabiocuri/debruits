# Data layout

```
data/
├── config.json                                    ← live config (seeded from src/config.json)
├── icons/
│   ├── icon1.png                                  ← stamp icon PNGs (PNG with transparency)
│   └── menina1.png
├── Series/
│   ├── ALGAS/ALGAS_1_upscaled.png                 ← source images per series
│   └── CONCHAS/CONCHAS_10_upscaled.png
├── Cover/
│   ├── ALGAS/ALGAS_6_upscaled.png                 ← cover photos per series
│   └── CONCHAS/CONCHAS_17_upscaled.png
├── Templates/
│   └── POSTCARD/
│       ├── PORTUGUES/
│       │   ├── ALGAS_icon.pdf                     ← saved print ({SERIES}_{ICON}.pdf)
│       │   ├── ALGAS_icon.json                    ← sidecar: layout, icon, lang …
│       │   └── ALGAS_icon_thumb.png
│       └── FRANCAIS/
│           └── PLANTAS_icon.pdf
└── Zines/
    └── FRUTAS_YYYYMMDD_HHMMSS.pdf                 ← saved zine PDF
```

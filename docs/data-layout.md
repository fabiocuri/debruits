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
│       │   ├── ALGAS_menina1.pdf                  ← saved print
│       │   ├── ALGAS_menina1.json                 ← sidecar: layout, icon, lang …
│       │   └── ALGAS_menina1_thumb.png
│       └── FRANCAIS/
│           └── PLANTAS_menina1.pdf
└── Zines/
    └── FRUTAS_20261125_210201.pdf                 ← saved zine PDF
```

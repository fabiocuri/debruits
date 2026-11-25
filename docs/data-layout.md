# Data layout

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

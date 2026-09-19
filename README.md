# DE BRUITS

DE BRUITS is an art project that combines microscopy and macro photography with AI image generation (GAN/CAN) to produce postcards, posters, and fanzines.

**Author:** Fabio Curi  
**Contact:** fcuri91@gmail.com

---

## Repository structure

```
debruits/
├── can/          Creative Adversarial Network — style-divergent image generation
├── gan/          Generative Adversarial Network — image synthesis and processing
├── frontend/     Legacy Vue.js frontend (archived)
└── webapp/       Project dashboard — template generation, image browser, interests tab
```

## Quick start

The webapp is the main daily interface. It starts both the DE BRUITS dashboard and the personal interests panel in one command:

```bash
cd webapp
pip install flask pillow fastapi "uvicorn[standard]" anthropic jinja2 python-dotenv icalendar recurring-ical-events markupsafe
python3 app.py
```

Open **http://localhost:5000**.

See [`webapp/README.md`](webapp/README.md) for full setup details.

## Image generation

### CAN (Creative Adversarial Network)
```bash
cd can
pip install -r requirements_can.txt
bash run_local_can.sh
```

### GAN
```bash
cd gan
pip install -r requirements_gan.txt
bash run_local_gan.sh
```

Generated images land in the `Final/` directory configured in `webapp/config.json` and appear automatically in the dashboard's Images tab.

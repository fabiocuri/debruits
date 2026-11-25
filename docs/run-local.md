# Run — local

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

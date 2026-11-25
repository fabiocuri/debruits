# Configuration

The live config is `data/config.json`. On first run the app seeds it from `src/config.json` (the default shipped in the image) if it doesn't exist yet.

| Key | Description |
|-----|-------------|
| `series_titles` | Per-series titles per language |
| `back_footer` | Postcard back footer text per language |
| `back_description` | Zine back description per language |
| `layout` | Stamp position, margins, line widths |
| `formats` | Canvas pixel dimensions per format |

`DATA_ROOT` env var sets the data directory. k8s sets it to `/data`; local default is `../data` relative to `src/`.

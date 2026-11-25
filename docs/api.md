# API

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Single-page app |
| GET | `/static/<file>` | CSS / JS assets |
| GET | `/img/<path>` | Serve image from DATA_ROOT |
| GET | `/font/<name>` | Custom TTF font |
| GET | `/icon/<name>` | Stamp icon PNG |
| GET/PUT | `/api/config` | Read / update config.json |
| GET | `/api/images` | List all Series/ source images |
| GET | `/api/covers/<series>` | List Cover/<series>/ photos |
| GET | `/api/icons` | List stamp icons |
| GET/PUT | `/api/thumbnails` | Per-series thumbnail overrides |
| GET | `/api/templates` | List saved prints |
| DELETE | `/api/templates` | Delete prints |
| GET | `/api/layout-data/<path>` | Sidecar JSON for a saved print |
| POST | `/api/generate` | Preview layout (no disk write) |
| POST | `/api/compose` | Render + save print PDF |
| POST | `/api/refresh-prints` | Re-render all prints from sidecars |
| GET | `/api/download/<path>` | Download one print PDF |
| GET | `/api/download-all` | Zip all print PDFs |
| GET | `/api/zines` | List saved zines |
| POST | `/api/zine/create` | Render + save zine PDF |
| DELETE | `/api/zines/delete` | Delete zines |
| POST | `/api/refresh-zines` | Re-render all zines from sidecars |
| GET | `/api/zines/download/<name>` | Download one zine PDF |
| GET | `/api/zines/download-all` | Zip all zine PDFs |

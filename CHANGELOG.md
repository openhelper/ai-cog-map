# Changelog

## 0.2.0 — 2026-03-15

### Added
- **Component metadata**: Hook now classifies each sub-module by layer index, component type (attention, mlp, projection, norm, etc.), and category. Writes a JSON metadata sidecar to `/dev/shm/aicogmap-metadata.json`.
- **Cognitive depth slider**: UI control to filter visible hooks by category — from layer outputs only (64 cells) to full detail (800+ cells).
- **Auto-interpreter**: Computes cognitive state labels (Attending, Reasoning, Structuring, Recalling, Searching, Generating, Processing) from activation energy patterns.
- **Component-aware rendering**: Heatmap cells are tinted by category — cyan for attention, pink for MLP, orange for projections, green for norms.
- **Energy breakdown**: Six-bar visualization showing attention/MLP/early/mid/deep/total energy.
- **Info tooltip**: Built-in component reference table and cognitive state explanation.
- **`GET /api/activations?depth=N`**: Depth parameter for server-side category filtering.
- **`GET /api/metadata`**: New endpoint returning full hook metadata.
- **`compute_cognitive_state()`**: New reader function for computing cognitive state from activation patterns.

### Changed
- Shared memory format version 1 → 2.
- `max_layers` default increased from 256 to 1024.
- Reader now accepts `meta_path` parameter and returns metadata alongside norms.
- Server health endpoint now reports metadata availability and version.
- License changed from MIT to Apache 2.0.

## 0.1.0 — 2026-03-14

### Added
- Initial release.
- SGLang forward hook plugin capturing per-layer activation norms.
- Shared memory bridge with lock-free binary format.
- Background flusher thread at 100ms intervals.
- Standalone FastAPI visualization server.
- Real-time heatmap UI with glow levels and delta bar.
- Docker Compose example.
- PyPI-ready packaging.

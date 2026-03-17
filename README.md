# AI Cog Map

Real-time transformer layer activation visualization for SGLang.

AI Cog Map hooks into SGLang's forward pass and captures per-layer activation norms, then renders them as a live heatmap. You can see which layers, attention heads, and MLP blocks are actually doing work on each token.

## Features

- Live heatmap grid where each cell is a hook point in the transformer. Colors go from dark (idle) through purple to bright cyan as activation intensity increases.
- Hook points are classified by component type — attention, MLP, projections, norms — with color tinting so you can visually distinguish the model's internal structure.
- **Cognitive depth slider** lets you control granularity. At level 1 you see 64 cells (one per layer). Crank it to 5 and you get every sub-module, 800+ cells.
- Auto-interpreter that computes a cognitive state from the activation patterns: Attending, Reasoning, Structuring, Recalling, Searching, Generating, Processing. It's a rough heuristic based on energy ratios between attention and MLP blocks, and where in the layer stack the activity is concentrated.
- Delta bar showing which layers changed since the last poll — useful for seeing where the model shifts attention as it works through a sequence.
- Stats: active layer count, mean activation, data age, energy breakdown by component type and layer depth.
- Built-in info tooltip explaining what each component type does.

Different tasks look different. Code generation has a distinct activation signature compared to creative writing. Math reasoning lights up different layers than summarization. Once you've watched it for a while you start to recognize the patterns.

## Architecture

```
┌─────────────────────┐    shared memory     ┌─────────────────────┐
│                     │    + metadata JSON    │                     │
│   SGLang Server     │  ─────────────────>   │   AI Cog Map UI     │
│                     │  /dev/shm/aicogmap-*  │                     │
│  + forward hooks    │                       │  FastAPI + HTML     │
│  (aicogmap.hook)    │                       │  localhost:7890     │
│                     │                       │                     │
└─────────────────────┘                       └─────────────────────┘
```

The **hook plugin** (`aicogmap.hook`) is a SGLang forward hook factory that attaches to every matched sub-module via `--forward-hooks`. It computes `output.detach().float().norm().item()` on each forward pass and writes to shared memory. It also classifies each hook point (layer index, component type, category) and writes a metadata sidecar JSON. No SGLang source modification needed.

The **visualization server** (`aicogmap.server`) is a standalone FastAPI app that reads the shared memory buffer and metadata, computes cognitive state, and serves the heatmap UI at 500ms poll intervals.

Communication between the two is through `/dev/shm` — a lock-free binary format (4-byte magic `ACGM` + header + float32 array) plus a JSON sidecar for metadata. Everything stays in RAM, no disk I/O involved.

## Installation

```bash
pip install aicogmap
```

Or from source:

```bash
git clone https://github.com/sonnyvleisides/ai-cog-map.git
cd ai-cog-map
pip install -e .
```

## Quick Start

### 1. Start SGLang with hooks enabled

```bash
python -m sglang.launch_server \
    --model-path Qwen/Qwen3.5-27B \
    --dtype bfloat16 \
    --port 8000 \
    --forward-hooks '[{
        "name": "aicogmap",
        "target_modules": ["model.layers.*"],
        "hook_factory": "aicogmap.hook:create_activation_hook",
        "config": {"shm_path": "/dev/shm/aicogmap-activations"}
    }]'
```

### 2. Start the visualization server

```bash
aicogmap --port 7890
```

### 3. Open the UI

Navigate to `http://localhost:7890` and send requests to your model. Watch the layers light up.

## Docker

If SGLang runs in Docker, mount shared memory and the aicogmap package:

```yaml
volumes:
  - /dev/shm:/dev/shm
  - ./aicogmap:/usr/local/lib/python3.12/dist-packages/aicogmap
```

See `examples/docker-compose.example.yml` for a complete setup.

## Configuration

### Hook Config

| Option | Default | Description |
|--------|---------|-------------|
| `shm_path` | `/dev/shm/aicogmap-activations` | Shared memory file path |
| `meta_path` | `/dev/shm/aicogmap-metadata.json` | Metadata sidecar JSON path |
| `flush_interval` | `0.1` | Seconds between shared memory flushes |
| `max_layers` | `1024` | Maximum hook points to track |

### Server Config

| Flag | Default | Description |
|------|---------|-------------|
| `--host` | `0.0.0.0` | Bind address |
| `--port` | `7890` | HTTP port |
| `--shm-path` | `/dev/shm/aicogmap-activations` | Shared memory path |
| `--meta-path` | `/dev/shm/aicogmap-metadata.json` | Metadata sidecar path |

## Target Module Patterns

The `target_modules` field uses `fnmatch` patterns. Common patterns:

| Pattern | Matches |
|---------|---------|
| `model.layers.*` | All transformer layers and sub-modules (recommended) |
| `model.layers.*.self_attn` | Attention modules only |
| `model.layers.*.mlp` | Feed-forward modules only |
| `model.layers.0` | First layer only |
| `model.layers.3[0-9]` | Layers 30-39 |

## Component Classification

When using `model.layers.*`, the hook automatically classifies each sub-module:

| Component | Category | What it captures |
|-----------|----------|------------------|
| `layer_output` | layer | Full layer contribution to residual stream |
| `attention` | attention | Attention block aggregated output |
| `attention_conv` | attention | Mamba-style convolution (hybrid models) |
| `attention_qkv` | attention | Query/Key/Value projections |
| `attention_gate` | attention | Attention gating mechanism |
| `mlp` | mlp | Feed-forward network block |
| `mlp_gate_up` | mlp | MLP gated projection (SwiGLU) |
| `mlp_down` | mlp | MLP output down-projection |
| `layernorm_input` | norm | Pre-attention normalization |
| `layernorm_post` | norm | Post-attention normalization |

The cognitive depth slider controls which categories are visible:

| Depth | Visible Categories | Typical cell count (64-layer model) |
|-------|-------------------|-------------------------------------|
| 1 | Layer outputs only | 64 |
| 2 | + Attention + MLP | ~192 |
| 3 | + Projections | ~448 |
| 4 | + Norms | ~576 |
| 5 | Everything | ~865 |

## Cognitive States

The auto-interpreter computes a cognitive state label from activation energy patterns:

| State | Pattern | Typical tasks |
|-------|---------|---------------|
| **Attending** | Attention energy > 65% of total | Reading context, following references |
| **Reasoning** | MLP dominant, deep layers active | Math, logic, code generation |
| **Structuring** | MLP dominant, early layers active | Parsing syntax, organizing output |
| **Recalling** | MLP gate projections highest | Factual recall, knowledge lookup |
| **Searching** | Low variance across layers | Uncertain, exploring possibilities |
| **Generating** | Deep layers > 50% of total energy | Token generation, text completion |
| **Processing** | No dominant pattern | Mixed or transitional processing |

## API

### `GET /api/activations?depth=N`

Returns normalized activation norms, deltas, cognitive state, and hook metadata.

```json
{
  "status": "ok",
  "num_layers": 192,
  "total_hooks": 865,
  "age_s": 0.05,
  "depth": 2,
  "norms": [0.23, 0.45, 0.12, "..."],
  "raw_norms": [145.2, 278.9, 72.1, "..."],
  "deltas": [0.01, 0.15, 0.03, "..."],
  "cognitive_state": {
    "label": "Reasoning",
    "confidence": 0.82,
    "attention_ratio": 0.31,
    "energy": {
      "attention": 1240.5,
      "mlp": 2890.3,
      "early": 980.2,
      "mid": 1650.8,
      "deep": 1499.8,
      "total": 4130.8
    }
  },
  "hooks": [
    {"index": 0, "layer": 0, "type": "layer_output", "category": "layer"},
    {"index": 1, "layer": 0, "type": "attention", "category": "attention"},
    "..."
  ],
  "history": ["..."]
}
```

### `GET /api/metadata`

Returns the full hook metadata sidecar.

### `GET /api/health`

Returns connection status, hook count, and metadata availability.

## Performance

Hook overhead is roughly 0.1-0.5% of inference time — one `tensor.norm()` per hook per forward pass. Shared memory footprint is about 4KB for 1024 hook points, plus another ~4KB one-time write for the metadata JSON. When hooks aren't enabled via `--forward-hooks`, overhead is zero. Everything goes through `/dev/shm` so there's no disk I/O.

## How It Works

SGLang's hook manager calls the hook function after each matched sub-module executes. The first time a module is seen, the hook resolves its fully qualified name (e.g. `model.layers.12.mlp.gate_up_proj`), classifies the component type, and appends to the metadata sidecar.

On every forward pass after that, it just computes `output.detach().float().norm().item()` and writes the scalar to a thread-safe buffer. A background flusher thread pushes the buffer to shared memory at configurable intervals (default 100ms).

The visualization server reads that shared memory, normalizes the values relative to the current max, runs the cognitive state heuristic, and serves it all to the frontend. The frontend maps normalized values to a color ramp and renders the grid.

L2 norm was chosen because it captures overall activation magnitude without being thrown off by individual outlier values, and it's cheap to compute on GPU tensors.

## Compatibility

Requires SGLang with `--forward-hooks` support (PR #13217), any transformer model SGLang can serve, any NVIDIA GPU, and Linux (needs `/dev/shm`).

## Author

Sonny Vleisides — [LinkedIn](https://www.linkedin.com/in/sonny-vleisides) · [ARRA Networks](https://arranetworks.com) · [GitHub](https://github.com/sonnyvleisides)

## License

Apache 2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE).

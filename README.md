# AI Cog Map

**Real-time transformer layer activation visualization for SGLang inference servers.**

Watch your AI model think. AI Cog Map captures per-layer activation patterns during inference and renders them as a live heatmap — showing which layers, attention heads, and MLP blocks are doing meaningful work on each forward pass.

## Features

- **Layer Activation Heatmap** — Grid where each cell represents a hook point in the transformer. Brightness shows activation intensity, from dark through purple to bright cyan.
- **Component-Aware Visualization** — Hook points are classified by type (attention, MLP, projections, norms) with distinct color tinting so you can see the model's internal structure.
- **Cognitive Depth Slider** — Control how much detail you see. Slide from "Layer Outputs" (64 cells for a 64-layer model) through to "Full Detail" (every sub-module, 800+ cells).
- **Auto-Interpreter** — Computes a cognitive state label from activation patterns: Attending, Reasoning, Structuring, Recalling, Searching, Generating, or Processing. Shows what kind of work the model is doing right now.
- **Activation Delta Bar** — Shows which layers are *changing* between forward passes, revealing where the model's attention is shifting as it processes new tokens.
- **Live Statistics** — Active layer count, mean activation, data freshness, energy breakdown by component type and layer depth.
- **Info Tooltip** — Built-in reference explaining what each component type does and what high activation means.

Different types of work create different visual signatures. Code generation activates different layers than creative writing. Math reasoning lights up different patterns than summarization.

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

**Hook Plugin** (`aicogmap.hook`): SGLang forward hook factory. Attaches to every matched sub-module via SGLang's native `--forward-hooks` system. Computes the L2 norm of each module's output tensor and writes to a shared memory buffer. Also classifies each hook point by layer index, component type, and category, writing a metadata sidecar JSON. No SGLang source modification required.

**Visualization Server** (`aicogmap.server`): Standalone FastAPI server that reads the shared memory buffer and metadata, computes cognitive state, and serves a real-time heatmap UI. Polls at 500ms for smooth animation.

**Shared Memory Bridge**: Lock-free binary format (4-byte magic `ACGM` + header + float32 array) at `/dev/shm/aicogmap-activations`. Metadata sidecar at `/dev/shm/aicogmap-metadata.json`. Zero disk I/O, zero serialization overhead. The hook writes, the server reads.

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

- **Hook overhead**: ~0.1-0.5% of inference time (one `tensor.norm()` per hook per forward pass)
- **Memory**: ~4KB shared memory for 1024 hook points
- **Metadata**: One-time JSON write per new hook (~4KB for 865 hooks)
- **Zero overhead when disabled**: Hooks are opt-in via `--forward-hooks`
- **No disk I/O**: Shared memory (`/dev/shm`) is RAM-backed

## How It Works

On each forward pass through the model:

1. SGLang's hook manager calls the hook function after each matched sub-module executes
2. On first encounter, the hook resolves the module's fully qualified name, classifies its component type, and writes the metadata sidecar
3. The hook computes `output.detach().float().norm().item()` — a single scalar capturing the module's activation magnitude
4. The scalar is written to a thread-safe buffer, indexed by hook position
5. A background thread flushes the buffer to shared memory at configurable intervals
6. The visualization server reads shared memory and metadata, normalizes values, computes cognitive state, and serves the UI
7. The frontend renders normalized values as a color-mapped heatmap grid with component tinting

The L2 norm captures overall activation magnitude without being sensitive to individual outlier values, and is extremely cheap to compute on GPU tensors.

## Compatibility

- **SGLang**: Any version with `--forward-hooks` support (PR #13217)
- **Models**: Any transformer-based model served by SGLang
- **GPU**: Any NVIDIA GPU supported by SGLang
- **OS**: Linux (requires `/dev/shm` for shared memory)

## Author

**Sonny Vleisides** — CTO at ARRA Networks | VP Development at Alice Research

- [LinkedIn](https://www.linkedin.com/in/sonny-vleisides)
- [ARRA Networks](https://arranetworks.com)
- [GitHub](https://github.com/sonnyvleisides)

## License

Apache 2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE).

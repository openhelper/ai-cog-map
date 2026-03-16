"""
AI Cog Map — Shared Memory Reader

Reads activation data written by the SGLang hook. No torch dependency.
Also reads the metadata sidecar for component type classification.
"""

import json
import struct
import time
from pathlib import Path
from typing import Any

DEFAULT_SHM_PATH = "/dev/shm/aicogmap-activations"
DEFAULT_META_PATH = "/dev/shm/aicogmap-metadata.json"
HEADER_SIZE = 64
MAGIC = b"ACGM"

DEPTH_LEVELS: dict[int, set[str]] = {
    1: {"layer"},
    2: {"layer", "attention", "mlp"},
    3: {"layer", "attention", "mlp", "projection"},
    4: {"layer", "attention", "mlp", "projection", "norm"},
    5: {"layer", "attention", "mlp", "projection", "norm", "other"},
}


def read_metadata(meta_path: str = DEFAULT_META_PATH) -> dict[str, Any] | None:
    """Read the hook metadata sidecar JSON. Returns parsed dict or None."""
    try:
        p = Path(meta_path)
        if not p.exists():
            return None
        with open(meta_path) as f:
            return json.load(f)
    except Exception:
        return None


def read_activations(
    shm_path: str = DEFAULT_SHM_PATH,
    meta_path: str = DEFAULT_META_PATH,
    depth: int = 0,
) -> dict[str, Any] | None:
    """
    Read current activation data from shared memory.

    Args:
        shm_path: Path to the shared memory activation file.
        meta_path: Path to the metadata sidecar JSON.
        depth: Cognitive depth filter (0 = all, 1-5 = filtered by category).

    Returns dict with keys: version, num_layers, total_hooks, timestamp_ns,
    age_s, norms, metadata (if available), or None if unavailable.
    """
    try:
        path = Path(shm_path)
        if not path.exists():
            return None

        with open(shm_path, "rb") as f:
            data = f.read()

        if len(data) < HEADER_SIZE:
            return None

        magic = data[:4]
        if magic != MAGIC:
            return None

        version = struct.unpack("<I", data[4:8])[0]
        num_layers = struct.unpack("<I", data[8:12])[0]
        timestamp_ns = struct.unpack("<Q", data[12:20])[0]

        if num_layers == 0:
            return None

        available = min(num_layers, (len(data) - HEADER_SIZE) // 4)
        if available == 0:
            return None

        norms: list[float] = []
        for i in range(available):
            offset = HEADER_SIZE + i * 4
            val = struct.unpack("<f", data[offset:offset + 4])[0]
            norms.append(val)

        age_s = (time.time_ns() - timestamp_ns) / 1e9

        meta = read_metadata(meta_path)
        hooks_meta: list[dict[str, Any]] | None = None
        if meta and "hooks" in meta:
            hooks_meta = meta["hooks"]

        if depth > 0 and hooks_meta and depth in DEPTH_LEVELS:
            allowed = DEPTH_LEVELS[depth]
            filtered_norms: list[float] = []
            filtered_meta: list[dict[str, Any]] = []
            for i, norm in enumerate(norms):
                hook_info = hooks_meta[i] if i < len(hooks_meta) else None
                cat = hook_info["category"] if hook_info else "other"
                if cat in allowed:
                    filtered_norms.append(norm)
                    if hook_info:
                        filtered_meta.append(hook_info)
            norms = filtered_norms
            hooks_meta = filtered_meta if filtered_meta else None

        result: dict[str, Any] = {
            "version": version,
            "num_layers": len(norms),
            "total_hooks": num_layers,
            "timestamp_ns": timestamp_ns,
            "age_s": round(age_s, 3),
            "norms": norms,
        }
        if hooks_meta:
            result["metadata"] = hooks_meta

        return result
    except Exception:
        return None


def compute_cognitive_state(
    norms: list[float],
    hooks_meta: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    """
    Compute the auto-interpreter cognitive state from activation patterns.

    Returns dict with label, ratios, and energy breakdown.
    """
    if not norms or not hooks_meta or len(norms) != len(hooks_meta):
        return {"label": "Idle", "confidence": 0}

    attn_energy = 0.0
    mlp_energy = 0.0
    mlp_gate_energy = 0.0
    total_energy = 0.0
    early_energy = 0.0   # layers 0-20
    mid_energy = 0.0     # layers 21-42
    deep_energy = 0.0    # layers 43-63
    layer_energies: list[float] = []

    for norm, meta in zip(norms, hooks_meta):
        cat = meta.get("category", "other")
        layer = meta.get("layer")
        comp_type = meta.get("type", "")

        total_energy += norm

        if cat in ("attention",):
            attn_energy += norm
        elif cat in ("mlp",):
            mlp_energy += norm
            if comp_type == "mlp_gate_up":
                mlp_gate_energy += norm

        if layer is not None:
            if layer <= 20:
                early_energy += norm
            elif layer <= 42:
                mid_energy += norm
            else:
                deep_energy += norm

        if comp_type == "layer_output" and layer is not None:
            layer_energies.append(norm)

    if total_energy == 0:
        return {"label": "Idle", "confidence": 0}

    combined = attn_energy + mlp_energy
    ratio = attn_energy / combined if combined > 0 else 0.5

    variance = 0.0
    if layer_energies:
        mean_le = sum(layer_energies) / len(layer_energies)
        variance = sum((e - mean_le) ** 2 for e in layer_energies) / len(layer_energies)

    label = "Processing"
    confidence = 0.5

    if ratio > 0.65:
        label = "Attending"
        confidence = min(1.0, ratio)
    elif ratio < 0.35 and deep_energy > mid_energy:
        label = "Reasoning"
        confidence = min(1.0, deep_energy / (mid_energy + 0.001))
    elif ratio < 0.35 and early_energy > deep_energy:
        label = "Structuring"
        confidence = min(1.0, early_energy / (deep_energy + 0.001))
    elif mlp_gate_energy > attn_energy:
        label = "Recalling"
        confidence = min(1.0, mlp_gate_energy / (attn_energy + 0.001))
    elif variance < 0.01 and layer_energies:
        label = "Searching"
        confidence = max(0.3, 1.0 - variance * 100)
    elif deep_energy > 0.5 * total_energy:
        label = "Generating"
        confidence = min(1.0, deep_energy / total_energy * 2)

    return {
        "label": label,
        "confidence": round(confidence, 3),
        "attention_ratio": round(ratio, 3),
        "energy": {
            "attention": round(attn_energy, 2),
            "mlp": round(mlp_energy, 2),
            "early": round(early_energy, 2),
            "mid": round(mid_energy, 2),
            "deep": round(deep_energy, 2),
            "total": round(total_energy, 2),
        },
    }

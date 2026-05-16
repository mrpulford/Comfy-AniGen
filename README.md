# Comfy-AniGen

ComfyUI plugin for [AniGen](https://arxiv.org/abs/2604.08746) — generate a rigged, skinned 3D character from a single image.

**No pytorch3d. No nvdiffrast.** Rasterization runs on Triton (bundled with PyTorch).

---

## Requirements

- Linux (Triton requires Linux)
- NVIDIA GPU with ≥ 18 GB VRAM
- ComfyUI with PyTorch 2.4+

---

## Installation

### Via ComfyUI Manager (recommended)

1. Open ComfyUI Manager → *Install via Git URL*
2. Paste the repo URL and click Install
3. Manager runs `install.py` automatically — this installs pip deps and trispconv
4. Restart ComfyUI

### Manual

```bash
# From your ComfyUI root:
cd custom_nodes
git clone <repo-url> Comfy-AniGen
cd Comfy-AniGen
python install.py
```

Then restart ComfyUI.

---

## Checkpoints

~10 GB of model weights are downloaded from HuggingFace (`VAST-AI/AniGen`) **on first use** into `Comfy-AniGen/ckpts/`. No manual download needed.

---

## Nodes

### AniGen Load Pipeline
Loads and caches the AniGen model. Connect its output to *AniGen Image → 3D*.

| Input | Default | Notes |
|---|---|---|
| `ss_flow_path` | `ckpts/anigen/ss_flow_duet` | Sparse structure flow model |
| `slat_flow_path` | `ckpts/anigen/slat_flow_auto` | Structured latent flow model |
| `use_ema` | false | Use EMA weights if available |

### AniGen Image → 3D
Runs the full pipeline: image → rigged GLB mesh + skeleton.

| Input | Default | Notes |
|---|---|---|
| `image` | — | ComfyUI IMAGE tensor |
| `seed` | 42 | |
| `cfg_scale_ss` | 7.5 | Guidance scale for sparse structure stage |
| `cfg_scale_slat` | 3.0 | Guidance scale for structured latent stage |
| `ss_steps` | 25 | Diffusion steps, sparse structure |
| `slat_steps` | 25 | Diffusion steps, structured latent |
| `joints_density` | 1 | Joint count level 0–4 (0 = minimal, 4 = dense) |
| `simplify_ratio` | 0.95 | Mesh decimation ratio (0 = no decimation) |
| `fill_holes` | true | Remove interior/invisible faces after decimation |
| `smooth_skin` | true | Laplacian smoothing of skinning weights |
| `smooth_iters` | 100 | Smoothing iterations |
| `texture_size` | 1024 | UV texture atlas resolution (0 = no texture) |

**Outputs:** `preview` image, `glb_path` (rigged mesh), `skeleton_path` (skeleton visualisation).

The GLB files are written to ComfyUI's output directory.

---

## Technical Details

See [TECHNICAL.md](TECHNICAL.md) for documentation of the Triton rasterizer, camera math, and texture baking pipeline.

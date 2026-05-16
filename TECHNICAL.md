# Comfy-AniGen — Technical Notes

This document explains the non-obvious implementation choices in the inference path: the Triton rasterizer, the camera math, the trispconv NaN fix, and the texture baking pipeline. The upstream AniGen paper and repo describe the model architecture; this covers only the engineering that differs from the original.

---

## 1. Why No pytorch3d or nvdiffrast

The original AniGen repo depends on two compiled CUDA extensions:

| Dependency | Used for | Problem |
|---|---|---|
| **pytorch3d** | KNN/ball-query ops, camera transforms, mesh rasterization | Non-trivial to build; requires matching CUDA/PyTorch versions; SONAME issues on some distros |
| **nvdiffrast** | Differentiable mesh rasterization for texture baking | Non-permissive license (NVIDIA Research, non-commercial) |

Both are eliminated in this plugin. The inference path uses:
- **Triton** (bundled with PyTorch on Linux) for rasterization
- **Pure-PyTorch** for camera transforms and KNN

---

## 2. Triton Rasterizer (`anigen/utils/triton_rasterizer.py`)

### What it replaces

`pytorch3d.renderer.mesh_renderer` / `rasterize_meshes` + `interpolate_face_attributes`.  
`nvdiffrast.torch.rasterize` + `nvdiffrast.torch.interpolate`.

### NDC convention

PyTorch3D NDC is **+X pointing left, +Y pointing up**, with (−1, −1) at the bottom-right pixel.  
Converting to screen pixels:

```
sx = (1 - ndc_x) / 2 * W      # +1 → column 0 (left),  −1 → column W (right)
sy = (1 - ndc_y) / 2 * H      # +1 → row 0 (top),      −1 → row H (bottom)
```

This matches pytorch3d's own NDC-to-screen mapping and is applied in `_ndc_to_screen` before the kernel runs.

### Kernel design

The kernel is a **pixel-parallel, face-sequential** rasterizer. The grid has `ceil(H*W / BLOCK_P)` programs; each handles `BLOCK_P` pixels in parallel and loops over all faces in chunks of `BLOCK_F`.

For each (pixel, face) pair the kernel:

1. Computes signed 2× triangle area and the three sub-triangle areas using the cross-product formula (2D version of the barycentric test):
   ```
   area  = (x1−x0)(y2−y0) − (x2−x0)(y1−y0)
   area0 = (x1−cx)(y2−cy) − (x2−cx)(y1−cy)   # sub-area opposite v0
   area1 = (x2−cx)(y0−cy) − (x0−cx)(y2−cy)   # sub-area opposite v1
   area2 = area − area0 − area1
   b_i   = area_i / area                        # barycentric coords
   ```
2. Tests coverage: `b0 >= −ε, b1 >= −ε, b2 >= −ε` (small ε=1e-5 for numerical tolerance at edges).
3. Interpolates depth: `z = b0·z0 + b1·z1 + b2·z2`.
4. Keeps the nearest (minimum z) face per pixel.

All conditionals are expressed with `tl.where` — no Python-level branching inside the kernel.

### The co-depth face collision bug (and fix)

When two faces share a pixel's minimum depth (degenerate case: adjacent coplanar triangles), a naive approach of `tl.sum(face_indices where at_min)` adds their indices — producing an **out-of-bounds face ID** (e.g. face 12000 + face 15000 = 27000, when F=20000). This caused CUDA index errors during texture baking.

The fix uses `tl.min` to deterministically pick the lowest-indexed winner among tied faces:

```python
# Among all faces at the block's minimum depth, find the one with the smallest index
min_win = tl.min(tl.where(at_min, f_idx_i,
                           tl.full([BLOCK_P, BLOCK_F], 2147483647, dtype=tl.int32)),
                 axis=1)
# Exactly one face per pixel satisfies the winning condition
is_best = at_min & (f_idx_i == tl.expand_dims(min_win, 1))
# Sum is now safe: exactly one True per row, so sum = that one value
win_face = tl.sum(tl.where(is_best, f_idx_i, zeros_pf_i), axis=1)
```

This guarantees a single deterministic winner even when faces share a depth value.

### `interpolate_face_attrs`

A pure-PyTorch function (no Triton kernel needed) that gathers per-face vertex attributes and blends them with barycentric weights:

```python
flat  = pix_to_face.clamp(min=0).reshape(-1)   # flatten, clamp background pixels
attrs = face_attrs[flat]                          # (N, 3, D) — gather face attrs
out   = (attrs * bary_coords.unsqueeze(-1)).sum(dim=-2)  # barycentric blend
```

Background pixels (pix_to_face < 0) are masked to −1.0 after blending.

---

## 3. Camera Math (`anigen/utils/geo_utils.py`)

### Why

`pytorch3d.renderer.cameras.FoVPerspectiveCameras` and `look_at_view_transform` use a **row-vector convention**: world points are row vectors, and the camera transform is `p_cam = p_world @ R + T`. This differs from the column-vector convention used by OpenGL and most other libraries.

### `_look_at_view_transform`

Constructs `R` and `T` matching pytorch3d's convention.

Given eye position, look-at target, and up vector:

```
z = normalize(at − eye)                  # forward axis (camera looks along +z)
x = normalize(cross(up, z))             # right axis
y = normalize(cross(z, x))             # corrected up axis
R = stack([x, y, z], dim=1)            # rows = camera axes, shape (3,3)
```

`R` is an orthonormal matrix whose rows are the camera's x/y/z axes expressed in world space.

The translation vector satisfies `p_cam = p_world @ R + T`, so for the eye itself:
```
0 = eye @ R + T   →   T = −(eye @ R)
```
But because `R` is orthonormal, `eye @ R = R.T @ eye` (column-vector equivalent), and the correct computation is:
```python
T = -(R @ eye.unsqueeze(-1)).squeeze(-1)
```

A subtle bug in an earlier version computed `T = -(eye @ R)` instead — this projects `eye` onto the *columns* of R rather than onto the *rows* (camera axes), giving a wrong translation whenever R is not symmetric. The result was every vertex projecting behind the camera, producing 100% invisible faces and an empty mesh.

### `_fov_perspective_project`

Projects world-space vertices into pytorch3d NDC:

```python
p_cam = verts @ R.T + T          # world → camera space (row-vector convention)
f = 1 / tan(fov / 2)             # focal length from field-of-view
ndc_x = f * p_cam[:, 0] / z      # perspective divide
ndc_y = f * p_cam[:, 1] / z
ndc_z = (far+near)/(far−near) − 2·far·near / ((far−near)·z)   # depth linearisation
```

This matches the NDC range and z convention that the Triton rasterizer expects.

### KNN / ball-query replacements

`knn_points` and `ball_query` from pytorch3d are replaced with pure-PyTorch implementations using `torch.cdist` for pairwise distances. The outputs are `NamedTuple` objects with the same field names (`.dists`, `.idx`, `.knn`) so call sites need no changes.

---

## 4. Texture Baking Pipeline

### Overview

After geometry is finalised, vertex colours from the SLAT decoder are baked into a UV texture atlas. The steps:

1. **UV parametrisation** — `xatlas` packs the mesh into a UV atlas (`parametrize_mesh`).
2. **Multiview rendering** — 100 camera views are rendered around the mesh using the Triton rasterizer, producing RGB observation images and camera matrices (`render_multiview_mesh_colors`).
3. **Texture baking** — observations are projected into UV space and blended into the atlas (`bake_texture`, `mode='fast'`).

### Why `render_multiview_mesh_colors` instead of nvdiffrast

The original path used `nvdiffrast.torch.rasterize` at `ssaa=4, resolution=1024`, which took ~5 minutes per generation on a 3090. The Triton-based renderer:

- Operates on the **decimated mesh** (~15k faces) rather than the raw decoded mesh (~450k faces)
- Uses `mode='fast'` (nearest-neighbour splat into UV space) instead of `mode='opt'` (TV-regularised optimisation)
- Achieves ~3 views/second, completing 100 views in ~34 seconds (~42× faster)

### `mode='fast'` vs `mode='opt'`

`mode='opt'` minimises a per-texel loss with total-variation regularisation (`lambda_tv`). Under-sampled regions (mesh extremities, sharp folds) receive a strong pull toward zero from the TV term, producing dark corners in the texture.

`mode='fast'` does a single forward pass: each observation pixel votes into the UV texel it covers, and contributions are averaged. No regularisation, no dark corners. Quality is slightly lower in regions with few views, but for 100 views at 1024 resolution this is imperceptible in practice.

---

## 5. Hole Filling (`_fill_holes` in `postprocessing_utils.py`)

After mesh decimation, interior cavities can appear where faces were removed. `_fill_holes` identifies them using multiview visibility:

1. Render the mesh from N viewpoints using the Triton rasterizer.
2. Collect all face IDs that appear in at least one render — these are *visible* faces.
3. Run a graph min-cut between visible and invisible faces; faces reachable only from the invisible side are interior and are removed.

A bounds guard (`face_id[(face_id >= 0) & (face_id < F)]`) is applied before the visibility count to protect against any stale rasterizer outputs, and the camera matrices are moved to the same device as the mesh vertices before use.

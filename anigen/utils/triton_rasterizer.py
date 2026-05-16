"""
Triton-based mesh rasterizer — drop-in for pytorch3d's rasterize_meshes +
interpolate_face_attributes in AniGen's inference path.

Public API (shapes match pytorch3d):
    rasterize_triangles(verts_ndc, faces, H, W)
        verts_ndc : (V, 3)  float32  — NDC coords, x/y in [-1,1], y-up
        faces     : (F, 3)  int32/int64
        returns   : pix_to_face (1,H,W,1) int64
                    zbuf        (1,H,W,1) float32
                    bary_coords (1,H,W,1,3) float32
                    dists       (1,H,W,1) float32   (zeros — blur_radius=0 only)

    interpolate_face_attrs(pix_to_face, bary_coords, face_attrs)
        pix_to_face : (1,H,W,K)
        bary_coords : (1,H,W,K,3)
        face_attrs  : (F,3,D)
        returns     : (1,H,W,K,D)
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# NDC → screen-pixel conversion (host side)
# pytorch3d NDC: x ∈ [-1,1] left→right, y ∈ [-1,1] bottom→top (y-up)
# screen: px ∈ [0,W], py ∈ [0,H], row 0 = top (y-down)
# ---------------------------------------------------------------------------

def _ndc_to_screen(verts_ndc: torch.Tensor, H: int, W: int):
    # pytorch3d NDC: +X points LEFT, +Y points UP → both use (1 - ndc) / 2 * size
    sx = (1.0 - verts_ndc[:, 0]) * 0.5 * W   # x: +1→0 (left), -1→W (right)
    sy = (1.0 - verts_ndc[:, 1]) * 0.5 * H   # y: +1→0 (top),  -1→H (bottom)
    return torch.stack([sx, sy, verts_ndc[:, 2]], dim=-1).contiguous()


# ---------------------------------------------------------------------------
# Triton kernel
#
# Grid: ceil(H*W / BLOCK_P) programs.
# Each program handles BLOCK_P pixels in parallel and iterates over all
# faces in chunks of BLOCK_F.  All conditionals use tl.where — no Python if.
# ---------------------------------------------------------------------------

@triton.jit
def _raster_kernel(
    x0_ptr, y0_ptr, z0_ptr,
    x1_ptr, y1_ptr, z1_ptr,
    x2_ptr, y2_ptr, z2_ptr,
    out_face_ptr,   # (H*W,) int32
    out_z_ptr,      # (H*W,) float32
    out_b0_ptr,     # (H*W,) float32
    out_b1_ptr,     # (H*W,) float32
    F, H, W,
    BLOCK_P: tl.constexpr,
    BLOCK_F: tl.constexpr,
):
    pid   = tl.program_id(0)
    p_off = tl.arange(0, BLOCK_P)              # (BLOCK_P,)
    p_idx = pid * BLOCK_P + p_off              # absolute pixel indices
    pmask = p_idx < H * W

    py = (p_idx // W).to(tl.float32)
    px = (p_idx %  W).to(tl.float32)
    cx = px + 0.5                              # pixel centre x  (BLOCK_P,)
    cy = py + 0.5                              # pixel centre y  (BLOCK_P,)

    INF = float('inf')
    best_z    = tl.full([BLOCK_P], INF,  dtype=tl.float32)
    best_face = tl.full([BLOCK_P], -1,   dtype=tl.int32)
    best_b0   = tl.zeros([BLOCK_P],      dtype=tl.float32)
    best_b1   = tl.zeros([BLOCK_P],      dtype=tl.float32)

    for f_start in range(0, F, BLOCK_F):
        f_idx = f_start + tl.arange(0, BLOCK_F)   # (BLOCK_F,)
        fmask = f_idx < F

        x0 = tl.load(x0_ptr + f_idx, mask=fmask, other=0.0)   # (BLOCK_F,)
        y0 = tl.load(y0_ptr + f_idx, mask=fmask, other=0.0)
        z0 = tl.load(z0_ptr + f_idx, mask=fmask, other=INF)
        x1 = tl.load(x1_ptr + f_idx, mask=fmask, other=0.0)
        y1 = tl.load(y1_ptr + f_idx, mask=fmask, other=0.0)
        z1 = tl.load(z1_ptr + f_idx, mask=fmask, other=INF)
        x2 = tl.load(x2_ptr + f_idx, mask=fmask, other=0.0)
        y2 = tl.load(y2_ptr + f_idx, mask=fmask, other=0.0)
        z2 = tl.load(z2_ptr + f_idx, mask=fmask, other=INF)

        # Expand for 2D (BLOCK_P, BLOCK_F) coverage test
        cx2 = tl.expand_dims(cx, 1)            # (BLOCK_P, 1)
        cy2 = tl.expand_dims(cy, 1)

        x0e = tl.expand_dims(x0, 0)            # (1, BLOCK_F)
        y0e = tl.expand_dims(y0, 0)
        z0e = tl.expand_dims(z0, 0)
        x1e = tl.expand_dims(x1, 0)
        y1e = tl.expand_dims(y1, 0)
        z1e = tl.expand_dims(z1, 0)
        x2e = tl.expand_dims(x2, 0)
        y2e = tl.expand_dims(y2, 0)
        z2e = tl.expand_dims(z2, 0)

        # Signed 2× areas: (BLOCK_P, BLOCK_F)
        area  = (x1e - x0e) * (y2e - y0e) - (x2e - x0e) * (y1e - y0e)
        area0 = (x1e - cx2) * (y2e - cy2) - (x2e - cx2) * (y1e - cy2)
        area1 = (x2e - cx2) * (y0e - cy2) - (x0e - cx2) * (y2e - cy2)
        area2 = area - area0 - area1

        valid    = (tl.abs(area) > 1e-8) & tl.expand_dims(fmask, 0)   # (BLOCK_P, BLOCK_F)
        inv_area = tl.where(valid, 1.0 / area, 0.0)

        b0 = area0 * inv_area                  # (BLOCK_P, BLOCK_F)
        b1 = area1 * inv_area
        b2 = area2 * inv_area

        EPS    = 1e-5
        inside = valid & (b0 >= -EPS) & (b1 >= -EPS) & (b2 >= -EPS)  # (BLOCK_P, BLOCK_F)

        z_int = b0 * z0e + b1 * z1e + b2 * z2e
        z_cmp = tl.where(inside, z_int, INF)   # (BLOCK_P, BLOCK_F)

        # Minimum depth per pixel across this face block: (BLOCK_P,)
        blk_min = tl.min(z_cmp, axis=1)

        # Among faces at the block minimum depth, pick the one with the lowest face index
        # as the sole winner.  Using tl.sum on f_idx when two faces share a pixel's minimum
        # depth produces an out-of-bounds face ID (e.g. 12000 + 15000 > F).
        do_upd  = (blk_min < best_z)                                      # (BLOCK_P,)
        at_min  = (z_cmp == tl.expand_dims(blk_min, 1)) \
                & tl.expand_dims(do_upd, 1) \
                & (tl.expand_dims(blk_min, 1) < INF)                      # (BLOCK_P, BLOCK_F)

        f_idx_i = tl.expand_dims(f_idx.to(tl.int32), 0)                  # (1, BLOCK_F)
        min_win = tl.min(tl.where(at_min, f_idx_i,
                                   tl.full([BLOCK_P, BLOCK_F], 2147483647, dtype=tl.int32)),
                         axis=1)                                           # (BLOCK_P,)
        is_best = at_min & (f_idx_i == tl.expand_dims(min_win, 1))       # exactly one True per row

        # Reduce winning face/bary to (BLOCK_P,) — exactly one face wins per pixel
        zeros_pf_i = tl.zeros([BLOCK_P, BLOCK_F], dtype=tl.int32)
        zeros_pf_f = tl.zeros([BLOCK_P, BLOCK_F], dtype=tl.float32)

        win_face = tl.sum(tl.where(is_best, f_idx_i, zeros_pf_i), axis=1)
        win_b0   = tl.sum(tl.where(is_best, b0, zeros_pf_f), axis=1)
        win_b1   = tl.sum(tl.where(is_best, b1, zeros_pf_f), axis=1)

        best_z    = tl.where(do_upd, blk_min, best_z)
        best_face = tl.where(do_upd, win_face, best_face)
        best_b0   = tl.where(do_upd, win_b0,  best_b0)
        best_b1   = tl.where(do_upd, win_b1,  best_b1)

    # Write outputs (background = -1 / -1.0)
    bg = (best_face < 0) | ~pmask
    tl.store(out_face_ptr + p_idx,
             tl.where(pmask, best_face, tl.full([BLOCK_P], -1, dtype=tl.int32)),
             mask=pmask)
    tl.store(out_z_ptr  + p_idx, tl.where(bg, tl.full([BLOCK_P], -1.0, dtype=tl.float32), best_z),   mask=pmask)
    tl.store(out_b0_ptr + p_idx, tl.where(bg, tl.full([BLOCK_P], -1.0, dtype=tl.float32), best_b0),  mask=pmask)
    tl.store(out_b1_ptr + p_idx, tl.where(bg, tl.full([BLOCK_P], -1.0, dtype=tl.float32), best_b1),  mask=pmask)


# ---------------------------------------------------------------------------
# Python wrapper
# ---------------------------------------------------------------------------

def rasterize_triangles(
    verts_ndc: torch.Tensor,   # (V, 3)  float32
    faces:     torch.Tensor,   # (F, 3)  int32 or int64
    H: int,
    W: int,
    BLOCK_P: int = 16,
    BLOCK_F: int = 64,
):
    """
    Returns (pix_to_face, zbuf, bary_coords, dists) — same shapes as pytorch3d.
      pix_to_face : (1,H,W,1) int64
      zbuf        : (1,H,W,1) float32
      bary_coords : (1,H,W,1,3) float32
      dists       : (1,H,W,1) float32  (zeros; blur_radius=0 only)
    """
    device = verts_ndc.device
    sc  = _ndc_to_screen(verts_ndc, H, W)     # (V, 3) screen coords

    f   = faces.long()
    v0  = sc[f[:, 0]]                         # (F, 3)
    v1  = sc[f[:, 1]]
    v2  = sc[f[:, 2]]

    def col(t, c): return t[:, c].contiguous().float()

    x0, y0, z0 = col(v0, 0), col(v0, 1), col(v0, 2)
    x1, y1, z1 = col(v1, 0), col(v1, 1), col(v1, 2)
    x2, y2, z2 = col(v2, 0), col(v2, 1), col(v2, 2)

    N = H * W
    out_face = torch.full((N,), -1,   dtype=torch.int32,   device=device)
    out_z    = torch.full((N,), -1.0, dtype=torch.float32, device=device)
    out_b0   = torch.full((N,), -1.0, dtype=torch.float32, device=device)
    out_b1   = torch.full((N,), -1.0, dtype=torch.float32, device=device)

    grid = (triton.cdiv(N, BLOCK_P),)
    _raster_kernel[grid](
        x0, y0, z0, x1, y1, z1, x2, y2, z2,
        out_face, out_z, out_b0, out_b1,
        faces.shape[0], H, W,
        BLOCK_P=BLOCK_P,
        BLOCK_F=BLOCK_F,
    )

    ptf  = out_face.view(1, H, W, 1).long()
    zbuf = out_z   .view(1, H, W, 1)
    b0   = out_b0  .view(1, H, W, 1)
    b1   = out_b1  .view(1, H, W, 1)
    b2   = torch.where(ptf >= 0, 1.0 - b0 - b1,
                       torch.full_like(b0, -1.0))
    bary  = torch.stack([b0, b1, b2], dim=-1)   # (1,H,W,1,3)
    dists = torch.zeros_like(zbuf)
    return ptf, zbuf, bary, dists


# ---------------------------------------------------------------------------
# interpolate_face_attrs — pure PyTorch, matches pytorch3d's API
# ---------------------------------------------------------------------------

def interpolate_face_attrs(
    pix_to_face: torch.Tensor,   # (1,H,W,K)
    bary_coords: torch.Tensor,   # (1,H,W,K,3)
    face_attrs:  torch.Tensor,   # (F,3,D)
) -> torch.Tensor:               # (1,H,W,K,D)
    flat  = pix_to_face.clamp(min=0).reshape(-1)            # (N,)
    attrs = face_attrs[flat]                                 # (N,3,D)
    attrs = attrs.reshape(*pix_to_face.shape, 3, face_attrs.shape[-1])  # (1,H,W,K,3,D)
    bary  = bary_coords.unsqueeze(-1)                        # (1,H,W,K,3,1)
    out   = (attrs * bary).sum(dim=-2)                       # (1,H,W,K,D)
    bg    = (pix_to_face < 0).unsqueeze(-1)
    return out.masked_fill(bg, -1.0)

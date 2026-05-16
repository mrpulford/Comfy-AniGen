"""Pure-PyTorch replacements for pytorch3d.ops.knn_points, ball_query,
and pytorch3d.renderer.cameras (look_at_view_transform / FoVPerspectiveCameras).

All functions match the pytorch3d API used by AniGen:
  - knn_points: returns namedtuple with .dists (squared L2), .idx, .knn=None
  - ball_query:  returns namedtuple with .dists (squared L2), .idx (-1 = empty slot), .knn=None
  - _look_at_view_transform: row-vector convention, same as pytorch3d
  - _fov_perspective_project: FoVPerspectiveCameras.transform_points_ndc equivalent
"""

import math
import torch
import torch.nn.functional as F
from typing import NamedTuple, Optional


class _KNNOutput(NamedTuple):
    dists: torch.Tensor   # (B, N, K) squared L2
    idx: torch.Tensor     # (B, N, K) int64
    knn: Optional[torch.Tensor]  # None (return_nn not implemented)


class _BallQueryOutput(NamedTuple):
    dists: torch.Tensor   # (B, N, K) squared L2, 0.0 for empty slots
    idx: torch.Tensor     # (B, N, K) int64, -1 for empty slots
    knn: Optional[torch.Tensor]


def knn_points(p1, p2, K=1, norm=2, return_nn=False):
    """
    p1: (B, N, D)  query points
    p2: (B, M, D)  reference points
    Returns _KNNOutput matching pytorch3d conventions.
    dists are squared L2 (norm=2) or L1 (norm=1, unsquared).
    """
    dists = torch.cdist(p1, p2, p=float(norm))   # (B, N, M)
    k = min(K, p2.shape[1])
    dists_k, idx_k = dists.topk(k, dim=-1, largest=False, sorted=True)
    if k < K:
        pad_d = dists_k.new_zeros((*dists_k.shape[:-1], K - k))
        pad_i = idx_k.new_full((*idx_k.shape[:-1], K - k), -1)
        dists_k = torch.cat([dists_k, pad_d], dim=-1)
        idx_k = torch.cat([idx_k, pad_i], dim=-1)
    if norm == 2:
        dists_k = dists_k ** 2
    return _KNNOutput(dists_k, idx_k, None)


def ball_query(p1, p2, K, radius):
    """
    p1: (B, N, D)  query points
    p2: (B, M, D)  reference points
    Returns _BallQueryOutput; slots beyond K or outside radius have idx=-1, dists=0.
    """
    dists_sq = torch.cdist(p1, p2) ** 2   # (B, N, M)
    in_radius = dists_sq <= (radius ** 2)

    B, N, M = dists_sq.shape
    masked = dists_sq.masked_fill(~in_radius, float('inf'))
    k = min(K, M)
    dists_k, idx_k = masked.topk(k, dim=-1, largest=False, sorted=True)

    if k < K:
        pad_d = dists_k.new_full((*dists_k.shape[:-1], K - k), float('inf'))
        pad_i = idx_k.new_full((*idx_k.shape[:-1], K - k), -1)
        dists_k = torch.cat([dists_k, pad_d], dim=-1)
        idx_k = torch.cat([idx_k, pad_i], dim=-1)

    oob = dists_k == float('inf')
    idx_k = idx_k.masked_fill(oob, -1)
    dists_k = dists_k.masked_fill(oob, 0.0)
    return _BallQueryOutput(dists_k, idx_k, None)


# ── Camera helpers (replaces FoVPerspectiveCameras + look_at_view_transform) ──

def _look_at_view_transform(eye, at, up):
    """
    Pure-PyTorch look_at matching pytorch3d row-vector convention.
    eye, at, up: (N, 3)  or  (3,) broadcast
    Returns R (N, 3, 3), T (N, 3).
    Transform: p_cam = p_world @ R + T
    """
    eye = torch.as_tensor(eye, dtype=torch.float32) if not isinstance(eye, torch.Tensor) else eye.float()
    at  = torch.as_tensor(at,  dtype=torch.float32) if not isinstance(at,  torch.Tensor) else at.float()
    up  = torch.as_tensor(up,  dtype=torch.float32) if not isinstance(up,  torch.Tensor) else up.float()

    z = F.normalize(at - eye, dim=-1)                               # (N, 3)
    x = F.normalize(torch.linalg.cross(up, z), dim=-1)             # (N, 3)
    # Degenerate: up ∥ z → fall back to x-axis as alternative up
    bad = torch.linalg.norm(x, dim=-1) < 1e-5                      # (N,)
    if bad.any():
        alt = torch.zeros_like(up)
        alt[..., 0] = 1.0
        x2 = F.normalize(torch.linalg.cross(alt, z), dim=-1)
        x = torch.where(bad.unsqueeze(-1), x2, x)
    y = F.normalize(torch.linalg.cross(z, x), dim=-1)              # (N, 3)
    R = torch.stack([x, y, z], dim=1)                              # (N, 3, 3), rows = camera axes
    T = -(R @ eye.unsqueeze(-1)).squeeze(-1)                       # (N, 3): T[j] = -(axis_j · eye)
    return R, T


def _fov_perspective_project(verts, R, T, fov_deg, znear, zfar):
    """
    Project world-space vertices to pytorch3d NDC using FoVPerspective convention.
    verts: (V, 3)   R: (3, 3)   T: (3,)
    Returns (V, 3) NDC coords suitable for _pt3d_rasterize.
    """
    p_cam = verts @ R.T + T                                    # (V, 3)
    z = p_cam[:, 2].clamp(min=znear)
    f = 1.0 / math.tan(fov_deg * math.pi / 360.0)
    ndc_x = f * p_cam[:, 0] / z
    ndc_y = f * p_cam[:, 1] / z
    ndc_z = (zfar + znear) / (zfar - znear) - 2.0 * zfar * znear / ((zfar - znear) * z)
    return torch.stack([ndc_x, ndc_y, ndc_z], dim=-1)

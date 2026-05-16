"""
ComfyUI nodes for AniGen — image → rigged 3D asset.

Depends on: triton, trimesh, xatlas, flash-attn (optional).
No pytorch3d or nvdiffrast required.
"""

import os
import sys
import gc

# Plugin root — ckpts/, anigen/ live here
_PLUGIN_DIR = os.path.dirname(os.path.abspath(__file__))

# Make the bundled anigen package importable
if _PLUGIN_DIR not in sys.path:
    sys.path.insert(0, _PLUGIN_DIR)


def _cuda_cleanup():
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ── Pipeline cache ─────────────────────────────────────────────────────────────

_pipeline_cache: dict = {}


def _load_pipeline(ss_flow_path: str, slat_flow_path: str, use_ema: bool):
    key = (ss_flow_path, slat_flow_path, use_ema)
    if key in _pipeline_cache:
        return _pipeline_cache[key]

    from anigen.pipelines import AnigenImageTo3DPipeline
    from anigen.utils.ckpt_utils import ensure_ckpts

    ensure_ckpts(_PLUGIN_DIR)
    pipeline = AnigenImageTo3DPipeline.from_pretrained(
        ss_flow_path=ss_flow_path,
        slat_flow_path=slat_flow_path,
        device="cuda",
        use_ema=use_ema,
    )
    pipeline.cuda()

    _pipeline_cache[key] = pipeline
    return pipeline


# ── Nodes ──────────────────────────────────────────────────────────────────────

class AniGenLoader:
    """Loads (and caches) the AniGen pipeline."""

    CATEGORY = "AniGen"
    FUNCTION = "load"
    RETURN_TYPES = ("ANIGEN_PIPELINE",)
    RETURN_NAMES = ("pipeline",)

    @classmethod
    def INPUT_TYPES(cls):
        ckpts = os.path.join(_PLUGIN_DIR, "ckpts", "anigen")
        return {
            "required": {
                "ss_flow_path":   ("STRING", {"default": os.path.join(ckpts, "ss_flow_duet")}),
                "slat_flow_path": ("STRING", {"default": os.path.join(ckpts, "slat_flow_auto")}),
                "use_ema":        ("BOOLEAN", {"default": False}),
            }
        }

    def load(self, ss_flow_path, slat_flow_path, use_ema):
        pipeline = _load_pipeline(ss_flow_path, slat_flow_path, use_ema)
        return (pipeline,)


class AniGenImageTo3D:
    """Runs AniGen: image → rigged GLB (mesh + skeleton + skinning weights)."""

    CATEGORY = "AniGen"
    FUNCTION = "generate"
    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("preview", "glb_path", "skeleton_path")
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline":       ("ANIGEN_PIPELINE",),
                "image":          ("IMAGE",),
                "seed":           ("INT",   {"default": 42,   "min": 0,   "max": 0xFFFFFFFF}),
                "cfg_scale_ss":   ("FLOAT", {"default": 7.5,  "min": 1.0, "max": 20.0, "step": 0.5}),
                "cfg_scale_slat": ("FLOAT", {"default": 3.0,  "min": 1.0, "max": 20.0, "step": 0.5}),
                "ss_steps":       ("INT",   {"default": 25,   "min": 10,  "max": 100}),
                "slat_steps":     ("INT",   {"default": 25,   "min": 10,  "max": 100}),
                "joints_density": ("INT",   {"default": 1,    "min": 0,   "max": 4}),
                "simplify_ratio": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.05}),
                "fill_holes":     ("BOOLEAN", {"default": True}),
                "smooth_skin":    ("BOOLEAN", {"default": True}),
                "smooth_iters":   ("INT",   {"default": 100,  "min": 1,   "max": 500}),
                "texture_size":   ("INT",   {"default": 1024, "min": 0,   "max": 2048, "step": 256}),
            }
        }

    def generate(
        self,
        pipeline,
        image,
        seed,
        cfg_scale_ss,
        cfg_scale_slat,
        ss_steps,
        slat_steps,
        joints_density,
        simplify_ratio,
        fill_holes,
        smooth_skin,
        smooth_iters,
        texture_size,
    ):
        import torch
        import numpy as np
        from PIL import Image as PILImage
        import folder_paths
        import time
        from comfy.utils import ProgressBar

        from anigen.utils.export_utils import convert_to_glb_from_data, visualize_skeleton_as_mesh

        # ComfyUI IMAGE tensor: (B, H, W, C) float32 [0,1] — take first frame
        img_np = (image[0].cpu().numpy() * 255).astype(np.uint8)
        pil_image = PILImage.fromarray(img_np)

        _PP_TICKS = 100
        pbar = ProgressBar(ss_steps + slat_steps + _PP_TICKS)

        def _ss_cb(step_idx, total_steps):
            pbar.update(1)

        def _slat_cb(step_idx, total_steps):
            pbar.update(1)

        _pp_done_ticks = [0]

        def _pp_cb(frac, desc):
            target = round(frac * _PP_TICKS)
            delta = target - _pp_done_ticks[0]
            if delta > 0:
                pbar.update(delta)
                _pp_done_ticks[0] = target

        with torch.no_grad():
            result = pipeline.run(
                image=pil_image,
                seed=seed,
                cfg_scale_ss=cfg_scale_ss,
                cfg_scale_slat=cfg_scale_slat,
                ss_steps=ss_steps,
                slat_steps=slat_steps,
                joints_density=joints_density,
                simplify_ratio=simplify_ratio,
                fill_holes=fill_holes,
                no_smooth_skin_weights=not smooth_skin,
                smooth_skin_weights_iters=smooth_iters,
                texture_size=texture_size,
                ss_progress_callback=_ss_cb,
                slat_progress_callback=_slat_cb,
                postprocess_progress_callback=_pp_cb,
            )

        output_dir = folder_paths.get_output_directory()
        stem = f"anigen_{int(time.time())}_{seed}"
        glb_path  = os.path.join(output_dir, f"{stem}.glb")
        skel_path = os.path.join(output_dir, f"{stem}_skeleton.glb")

        convert_to_glb_from_data(
            result['mesh'],
            result['joints'],
            result['parents'],
            result['skin_weights'],
            glb_path,
            vertex_colors=result.get('vertex_colors'),
            texture_image=result.get('texture_image'),
        )

        skel_mesh = result.get('skeleton_mesh')
        if skel_mesh is not None and len(skel_mesh.vertices) > 0:
            skel_mesh.export(skel_path)
        else:
            skel_path = ""

        processed = result['processed_image']
        preview_np = np.array(processed).astype(np.float32) / 255.0
        preview_tensor = torch.from_numpy(preview_np).unsqueeze(0)  # (1,H,W,C)

        return (preview_tensor, glb_path, skel_path)


NODE_CLASS_MAPPINGS = {
    "AniGenLoader":    AniGenLoader,
    "AniGenImageTo3D": AniGenImageTo3D,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AniGenLoader":    "AniGen Load Pipeline",
    "AniGenImageTo3D": "AniGen Image → 3D",
}

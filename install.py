"""
ComfyUI custom node installer — run automatically by ComfyUI Manager.
Can also be run manually: python install.py
"""

import subprocess
import sys
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_PY  = sys.executable


def _pip(*args):
    subprocess.check_call([_PY, "-m", "pip", "install", *args])


def _pip_quiet(*args):
    subprocess.check_call([_PY, "-m", "pip", "install", "-q", *args])


def main():
    print("=== Comfy-AniGen: installing dependencies ===")

    # 1. Core requirements
    req = os.path.join(_HERE, "requirements.txt")
    print(f"\n[1/2] Installing requirements from {req} ...")
    _pip("-r", req)

    # 2. trispconv — pure-Triton sparse 3D conv (replaces spconv / nvdiffrast)
    #    Triton ships with PyTorch on Linux, so no extra CUDA wheel needed.
    print("\n[2/2] Installing trispconv ...")
    _pip_quiet(
        "trispconv[triton] @ git+https://github.com/mrpulford/trispconv.git",
        "--upgrade",
    )

    print("\n=== Comfy-AniGen: installation complete ===")
    print("Checkpoints (~10 GB) will be downloaded from HuggingFace on first use.")


if __name__ == "__main__":
    main()

import os


HF_REPO_ID = "VAST-AI/AniGen"
CKPTS_DIR = "ckpts"

# Each entry: (local sentinel path relative to ckpts/, HF pattern to download if missing)
_REQUIRED = [
    ("dinov2/hubconf.py",          "ckpts/dinov2/**"),
    ("anigen/ss_flow_duet",        "ckpts/anigen/ss_flow_duet/**"),
    ("anigen/slat_flow_auto",      "ckpts/anigen/slat_flow_auto/**"),
]


def ensure_ckpts(local_dir: str = ".") -> None:
    """
    Check each required checkpoint piece individually and download only
    what is missing from HuggingFace (VAST-AI/AniGen).
    """
    from huggingface_hub import snapshot_download

    ckpts_path = os.path.join(local_dir, CKPTS_DIR)
    missing = [
        (sentinel, pattern)
        for sentinel, pattern in _REQUIRED
        if not os.path.exists(os.path.join(ckpts_path, sentinel))
    ]

    if not missing:
        return

    for sentinel, pattern in missing:
        print(f"Missing checkpoint '{sentinel}' — downloading from HuggingFace ({HF_REPO_ID}) ...")
        snapshot_download(
            repo_id=HF_REPO_ID,
            allow_patterns=[pattern],
            local_dir=local_dir,
        )

    print("Checkpoint download complete.")

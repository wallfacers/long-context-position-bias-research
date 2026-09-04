#!/usr/bin/env python3
"""Upload the staged HF release folder to the private dataset repo.

Reads HF_TOKEN from the environment. Uses upload_large_folder for
resumable, chunked upload of multi-GB trees.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from huggingface_hub import HfApi

REPO_ID = "wallfacers/position-bias-strict-block96"
FOLDER = Path(__file__).resolve().parents[1] / "artifacts" / "hf-release"


def main() -> int:
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("HF_TOKEN is not set", file=sys.stderr)
        return 2
    if not (FOLDER / "adapters").is_dir():
        print(f"Staging folder incomplete: {FOLDER}", file=sys.stderr)
        return 2
    api = HfApi(token=token)
    api.upload_large_folder(
        repo_id=REPO_ID,
        repo_type="dataset",
        folder_path=str(FOLDER),
    )
    print("upload complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

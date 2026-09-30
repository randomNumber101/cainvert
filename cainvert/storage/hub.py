"""Hugging Face Hub Integration for CAInvert Catalogs.

Provides append-only upload synchronization, live catalog discovery, and seamless
Parquet dataset retrieval from the Hugging Face Hub.
"""

import json
import os
import urllib.request
from typing import Any, Dict, List, Optional

try:
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download
    _HF_AVAILABLE = True
except ImportError:
    _HF_AVAILABLE = False


def check_hf_available() -> None:
    if not _HF_AVAILABLE:
        raise ImportError(
            "huggingface_hub is required for Hub operations. "
            "Please install it with: pip install huggingface_hub"
        )


def get_hub_catalog_manifest(
    repo_id: str = "randomNumber101/cainvert-benchmark",
    token: Optional[str] = None,
) -> Dict[str, Any]:
    """Retrieves the global inventory and metadata manifest from the Hugging Face dataset repo."""
    url = f"https://huggingface.co/datasets/{repo_id}/raw/main/CATALOG_MANIFEST.json"
    req = urllib.request.Request(url)
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req) as response:
            return json.loads(response.read().decode("utf-8"))
    except Exception as e:
        return {
            "error": f"Could not fetch Hub manifest from {url}: {e}",
            "repo_id": repo_id,
        }


def download_catalog_from_hub(
    repo_id: str = "randomNumber101/cainvert-benchmark",
    split: str = "1d_w81",
    local_dir: Optional[str] = None,
    token: Optional[str] = None,
) -> str:
    """Downloads or caches a specific catalog split from Hugging Face Hub.

    Returns:
        The local path containing the downloaded Parquet shards.
    """
    check_hf_available()
    if local_dir is None:
        local_dir = os.path.join("data", "catalogs", f"catalog_{split}")
    os.makedirs(local_dir, exist_ok=True)

    print(f"Fetching '{split}' from Hugging Face Hub ({repo_id}) into {local_dir}...")
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns=f"data/{split}/*.parquet",
        local_dir=local_dir,
        token=token,
    )
    return os.path.join(local_dir, "data", split)


def push_catalog_to_hub(
    local_catalog_dir: str,
    repo_id: str = "randomNumber101/cainvert-benchmark",
    split_name: str = "1d_w81",
    token: Optional[str] = None,
) -> None:
    """Appends all Parquet shards from a local catalog directory to the Hugging Face dataset repo.

    Because Parquet shards use collision-free, content-addressed names, appending new
    shards does not overwrite or corrupt previously mined data.
    """
    check_hf_available()
    api = HfApi(token=token)

    # Ensure dataset repo exists
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)

    print(f"Uploading Parquet shards from {local_catalog_dir} to {repo_id} (path: data/{split_name}/)...")
    api.upload_folder(
        folder_path=local_catalog_dir,
        path_in_repo=f"data/{split_name}",
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns="*.parquet",
    )
    print(f"Successfully uploaded and appended shards to Hugging Face Hub ({repo_id})!")

"""
Resumable high-speed upload of checkpoint directory to Hugging Face Hub.

Usage:
  Option A (Make repo public - NO storage limit, full 247GB):
    python scripts/upload_to_hf.py --repo-id qtris123/KimiK3 --public

  Option B (Keep repo private - upload key final checkpoints ~8GB):
    python scripts/upload_to_hf.py --repo-id qtris123/KimiK3 --key-only
"""

import argparse
import os
import sys
from pathlib import Path

# Enable high performance transfer
os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

try:
    from huggingface_hub import HfApi, create_repo
except ImportError as e:
    print(f"Error importing huggingface_hub: {e}")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Upload folder to Hugging Face Model Hub")
    parser.add_argument("--repo-id", required=True, help="HF Repo ID, e.g. 'qtris123/KimiK3'")
    parser.add_argument("--folder", default="/home/gpuuser/gstar-kimi3/out", help="Local folder to upload")
    parser.add_argument("--path-in-repo", default="KimiK3", help="Destination path inside the repo")
    parser.add_argument("--token", default=None, help="Hugging Face access token (or set HF_TOKEN env var)")
    parser.add_argument("--public", action="store_true", default=False, help="Make repository public (public repos have NO storage limit)")
    parser.add_argument("--key-only", action="store_true", default=False, help="Upload only key/final checkpoints (~8GB) to stay within private quota")
    parser.add_argument("--repo-type", default="model", choices=["model", "dataset"])
    args = parser.parse_args()

    token = args.token or os.environ.get("HF_TOKEN")
    if not token or token == "hf_your_write_token_here":
        print("\n[ERROR] No valid Hugging Face token provided!")
        print("Please export HF_TOKEN in your terminal or bashrc.")
        sys.exit(1)

    api = HfApi(token=token)

    # Validate authentication
    try:
        user_info = api.whoami()
        print(f"Authenticated as Hugging Face user: {user_info.get('name')}")
    except Exception as e:
        print(f"\n[ERROR] Failed to authenticate with Hugging Face: {e}")
        sys.exit(1)

    local_path = Path(args.folder).resolve()
    if not local_path.exists():
        print(f"Error: Local path {local_path} does not exist.")
        sys.exit(1)

    # Determine visibility
    is_private = not args.public
    visibility_str = "PUBLIC" if args.public else "PRIVATE"

    print(f"==================================================")
    print(f"Target HF Repo:   {args.repo_id} ({args.repo_type}, {visibility_str})")
    print(f"Local Folder:     {local_path}")
    print(f"Path in Repo:     {args.path_in_repo}")
    if args.key_only:
        print(f"Filter Mode:      Key checkpoints only (*019073*, *draft*)")
    else:
        print(f"Filter Mode:      Full directory (~247 GB)")
    print(f"==================================================")

    # 1. Create repo or update visibility
    try:
        create_repo(
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            token=token,
            private=is_private,
            exist_ok=True,
        )
        if args.public:
            try:
                api.update_repo_visibility(repo_id=args.repo_id, repo_type=args.repo_type, private=False)
                print(f"Repository {args.repo_id} visibility set to PUBLIC.")
            except Exception as e:
                print(f"Visibility update note: {e}")
        print(f"Repository {args.repo_id} ready.")
    except Exception as e:
        print(f"Repo setup note: {e}")

    # Patterns for key-only
    allow_patterns = None
    if args.key_only:
        # Match final checkpoint 019073 and draft models
        allow_patterns = ["*019073*", "*draft*"]

    # 2. Upload folder
    print("Starting upload (hashing files and uploading chunks)...")
    api.upload_folder(
        folder_path=str(local_path),
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        token=token,
        path_in_repo=args.path_in_repo,
        allow_patterns=allow_patterns,
        commit_message=f"Upload {local_path.name} checkpoints to {args.path_in_repo}",
    )
    print("\nUpload completed successfully!")


if __name__ == "__main__":
    main()

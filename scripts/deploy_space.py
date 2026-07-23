"""M5 pivot: create/update the public Gradio+ZeroGPU HF Space and push the
curated runtime file set (src/, data/metadata/store.db, requirements.txt,
README_GRADIO.md -> README.md). Idempotent -- safe to re-run after code
changes. Secrets are read from the local .env and pushed as Space Secrets;
values are never printed. Usage: python scripts/deploy_space.py [<ns>/<name>]"""
import shutil
import sys
import tempfile
from pathlib import Path

from dotenv import dotenv_values
from huggingface_hub import HfApi, SpaceHardware

REPO_ID_DEFAULT = "quockhanh41/vietnamese-legal-bot"
SECRET_KEYS = [
    "QDRANT_URL", "QDRANT_API_KEY", "GROQ_API_KEY", "GROQ_API_KEYS",
    "GEMINI_API_KEY", "GEMINI_API_KEYS", "OPENROUTER_API_KEY", "OPENROUTER_API_KEYS",
    "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_HOST", "REDIS_URL",
]
ROOT = Path(__file__).resolve().parent.parent


def stage_files(staging_dir: Path) -> None:
    shutil.copytree(ROOT / "src", staging_dir / "src",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (staging_dir / "data" / "metadata").mkdir(parents=True)
    shutil.copy(ROOT / "data/metadata/store.db", staging_dir / "data/metadata/store.db")
    shutil.copy(ROOT / "requirements.txt", staging_dir / "requirements.txt")
    shutil.copy(ROOT / "README_GRADIO.md", staging_dir / "README.md")


def push_secrets(api: HfApi, repo_id: str) -> None:
    values = dotenv_values(ROOT / ".env")
    pushed = []
    for key in SECRET_KEYS:
        val = values.get(key)
        if val:
            api.add_space_secret(repo_id, key, val)
            pushed.append(key)
    print(f"Secrets pushed (values not shown): {pushed}")


def main():
    repo_id = sys.argv[1] if len(sys.argv) > 1 else REPO_ID_DEFAULT
    api = HfApi()
    api.create_repo(repo_id, repo_type="space", space_sdk="gradio",
                    space_hardware=SpaceHardware.ZERO_A10G, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        staging = Path(tmp) / "space"
        staging.mkdir()
        stage_files(staging)
        api.upload_folder(repo_id=repo_id, repo_type="space", folder_path=str(staging),
                          commit_message="deploy: sync Gradio Space from repo")
    push_secrets(api, repo_id)
    print(f"Deployed: https://huggingface.co/spaces/{repo_id}")


if __name__ == "__main__":
    main()

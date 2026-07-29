#!/usr/bin/env python3
import sys

from huggingface_hub import HfApi


def deploy_warrant():
    print("=========================================")
    print("Warrant - Hugging Face Spaces Deploy")
    print("=========================================\n")

    hf_token = input("Enter your Hugging Face Access Token (with write permission): ").strip()
    if not hf_token:
        print("HF Token is required!")
        sys.exit(1)

    groq_key = input("Enter your Groq API Key: ").strip()
    if not groq_key:
        print("Groq API Key is required for the LLM verification step!")
        sys.exit(1)

    api = HfApi(token=hf_token)
    user = api.whoami()
    username = user["name"]

    space_name = "warrant"
    repo_id = f"{username}/{space_name}"

    print(f"\n-> Creating Space '{repo_id}' on Hugging Face (Docker Space)...")
    try:
        api.create_repo(repo_id=repo_id, repo_type="space", space_sdk="docker", private=False)
        print("Space created successfully.")
    except Exception as e:
        if "already exists" in str(e).lower():
            print("Space already exists. Proceeding with upload...")
        else:
            print(f"Error creating space: {e}")
            sys.exit(1)

    print("-> Adding GROQ_API_KEY to Space secrets...")
    api.add_space_secret(repo_id=repo_id, key="GROQ_API_KEY", value=groq_key)

    print("-> Uploading files (this might take a minute)...")
    api.upload_folder(
        folder_path=".",
        repo_id=repo_id,
        repo_type="space",
        ignore_patterns=[
            ".git",
            ".github",
            "__pycache__",
            "venv",
            ".venv",
            "tests",
            "bench",
            "*.pyc",
        ],
    )

    print("\n✅ Deployment triggered successfully!")
    print(f"Watch the build progress here: https://huggingface.co/spaces/{repo_id}")


if __name__ == "__main__":
    deploy_warrant()

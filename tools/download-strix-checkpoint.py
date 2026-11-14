"""Resume a pinned original DeepSeek Flash download without changing installed models."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

REPO = "deepseek-ai/DeepSeek-V4.1-Flash"
REVISION = "314113f8a7dadb388215396ba5fbee2c610a04d6"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        parser.error("workers must be between 1 and 8")
    args.directory.mkdir(parents=True, exist_ok=True)
    info = HfApi().model_info(REPO, revision=REVISION, files_metadata=True)
    # Keep original weights and tokenizer/config; do not execute remote Python.
    files = [f for f in info.siblings if f.rfilename.endswith(".safetensors")
             or f.rfilename in {"config.json", "inference/config.json", "tokenizer.json",
                                "tokenizer_config.json", "model.safetensors.index.json", "LICENSE"}]
    required = sum(f.size or 0 for f in files)
    complete = sum((args.directory / f.rfilename).stat().st_size for f in files
                   if (args.directory / f.rfilename).is_file())
    free = shutil.disk_usage(args.directory).free
    if free < max(0, required - complete) + 10 * 1024**3:
        raise RuntimeError(f"Insufficient disk: need up to {required-complete} more bytes plus 10 GiB reserve; free={free}")
    manifest = {"repo": REPO, "revision": REVISION, "expected_bytes": required,
                "files": [{"path": f.rfilename, "bytes": f.size,
                           "sha256": f.lfs.sha256 if f.lfs else None} for f in files]}
    (args.directory / "source-manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"destination": str(args.directory.resolve()), "expected_bytes": required,
                      "revision": REVISION}), flush=True)
    snapshot_download(REPO, revision=REVISION, local_dir=args.directory,
                      allow_patterns=[f.rfilename for f in files], max_workers=args.workers)
    missing = [f.rfilename for f in files if not (args.directory / f.rfilename).is_file()
               or (args.directory / f.rfilename).stat().st_size != f.size]
    if missing:
        raise RuntimeError(f"Download incomplete or incorrect lengths: {missing}")
    for entry in manifest["files"]:
        if entry["sha256"]:
            with (args.directory / entry["path"]).open("rb") as stream:
                actual = hashlib.file_digest(stream, "sha256").hexdigest()
            if actual != entry["sha256"]:
                raise RuntimeError(f"SHA256 mismatch: {entry['path']}")
            print(f"Verified SHA256: {entry['path']}", flush=True)
    (args.directory / "download-verified.json").write_text(
        json.dumps({"revision": REVISION, "files": len(files), "lfs_sha256_verified": True}, indent=2), encoding="utf-8")
    print("Download complete; expected file lengths and all LFS SHA256 hashes verified.", flush=True)


if __name__ == "__main__":
    main()

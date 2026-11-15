#!/usr/bin/env python3
"""Expand the exact ROCm 7.14 development wheel in target, reusing installed runtime files.

No package installation, vendor metadata edits, global PATH changes, or driver changes.
The wheel's references to installed core/library files become copies in our isolated
tree; file contents in the managed environment are only read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
NAME = "rocm_sdk_devel-7.14.0-py3-none-win_amd64.whl"
URL = f"https://repo.amd.com/rocm/whl-multi-arch/{NAME}"
SHA256 = "9d28c9c67f719869b73cc5fbf6d5dbe44fcf17f3aab5e37c4a7efa0b59c153af"
SIZE = 655223583


def digest(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def inside(path: Path, base: Path) -> Path:
    path = path.resolve()
    if not path.is_relative_to(base.resolve()):
        raise ValueError(f"Archive path escapes isolated destination: {path}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-site", type=Path, default=Path.home() / ".unsloth/studio/unsloth_studio/Lib/site-packages")
    args = parser.parse_args()
    work = ROOT / "target/qwen-resident"
    destination = work / "rocm-7.14"
    sdk = destination / "_rocm_sdk_devel"
    receipt_path = destination / "receipt.json"
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if receipt["wheel_sha256"] != SHA256 or receipt["runtime_site"] != str(args.runtime_site.resolve()):
            raise ValueError("Existing isolated SDK receipt differs; preserve it and inspect manually")
        if (sdk / "lib/cmake/hip/hip-config.cmake").exists():
            print(str(sdk))
            return
    for package in ("rocm_sdk_core", "rocm_sdk_libraries"):
        metadata = args.runtime_site / f"{package}-7.14.0.dist-info/METADATA"
        if not metadata.is_file() or "Version: 7.14.0\n" not in metadata.read_text():
            raise ValueError(f"Exact installed 7.14.0 runtime required: {metadata}")
    wheel = work / "downloads" / NAME
    wheel.parent.mkdir(parents=True, exist_ok=True)
    if not wheel.exists():
        partial = wheel.with_suffix(".part")
        urllib.request.urlretrieve(URL, partial)
        if partial.stat().st_size != SIZE or digest(partial) != SHA256:
            raise ValueError("ROCm development wheel checksum mismatch")
        partial.replace(wheel)
    if wheel.stat().st_size != SIZE or digest(wheel) != SHA256:
        raise ValueError("ROCm development wheel checksum mismatch")
    destination.mkdir(parents=True, exist_ok=True)
    pending: list[tuple[Path, Path]] = []
    with zipfile.ZipFile(wheel) as archive, archive.open("rocm_sdk_devel/_devel.tar") as payload:
        with tarfile.open(fileobj=payload, mode="r|") as tar:
            for member in tar:
                relative = PurePosixPath(member.name.replace("\\", "/"))
                output = inside(destination / relative, sdk)
                if member.isdir():
                    output.mkdir(parents=True, exist_ok=True)
                elif member.isfile():
                    output.parent.mkdir(parents=True, exist_ok=True)
                    with tar.extractfile(member) as source, output.open("wb") as target:
                        shutil.copyfileobj(source, target)
                elif member.issym():
                    link = Path(member.linkname.replace("\\", "/"))
                    local_target = (output.parent / link).resolve()
                    if local_target.is_relative_to(sdk.resolve()):
                        pending.append((output, local_target))
                    else:
                        relative_target = local_target.relative_to(destination.resolve())
                        if relative_target.parts[0] not in ("_rocm_sdk_core", "_rocm_sdk_libraries"):
                            raise ValueError(f"Unexpected wheel reference: {member.linkname}")
                        pending.append((output, args.runtime_site / relative_target))
                else:
                    raise ValueError(f"Unsupported archive entry: {member.name}")
    while pending:
        remaining = []
        for output, source in pending:
            if not source.is_file():
                remaining.append((output, source))
                continue
            output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, output)
        if len(remaining) == len(pending):
            raise ValueError(f"Unresolved wheel references: {remaining[:5]}")
        pending = remaining
    # The per-device runtime wheel has an explicit list of its additional files.
    # Copy those without invoking vendor init, which would rewrite its RECORD.
    manifest = args.runtime_site / "_rocm_sdk_libraries/.devel_links/gfx1151.json"
    for link in json.loads(manifest.read_text())["links"]:
        output = inside(sdk / link["relpath"], sdk)
        source = (args.runtime_site / "_rocm_sdk_devel" / Path(link["relpath"]).parent / link["target"]).resolve()
        inside(source, args.runtime_site / "_rocm_sdk_libraries")
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, output)
    receipt_path.write_text(json.dumps({"wheel_url": URL, "wheel_sha256": SHA256, "runtime_site": str(args.runtime_site.resolve()), "sdk": str(sdk)}, indent=2) + "\n")
    print(str(sdk))


if __name__ == "__main__":
    main()

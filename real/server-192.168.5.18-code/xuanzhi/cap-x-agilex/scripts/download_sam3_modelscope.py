#!/usr/bin/env python3
"""Download the official SAM3 checkpoint from ModelScope in parallel."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import os
import time
from pathlib import Path

import requests


URL = "https://www.modelscope.cn/models/facebook/sam3/resolve/master/sam3.pt"
SIZE = 3_450_062_241
SHA256 = "9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e"


def download_range(fd: int, start: int, end: int, retries: int = 8) -> int:
    expected = end - start + 1
    for attempt in range(retries):
        try:
            response = requests.get(
                URL,
                headers={"Range": f"bytes={start}-{end}"},
                timeout=(15, 120),
            )
            response.raise_for_status()
            data = response.content
            if response.status_code != 206 or len(data) != expected:
                raise RuntimeError(
                    f"range {start}-{end}: HTTP {response.status_code}, "
                    f"received {len(data)} of {expected} bytes"
                )
            os.pwrite(fd, data, start)
            return expected
        except Exception:
            if attempt + 1 == retries:
                raise
            time.sleep(min(2**attempt, 30))
    raise AssertionError("unreachable")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("checkpoints/sam3/sam3.pt"),
    )
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--chunk-mib", type=int, default=32)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".pt.parallel.incomplete")
    chunk_size = args.chunk_mib * 1024 * 1024
    ranges = [
        (start, min(start + chunk_size, SIZE) - 1)
        for start in range(0, SIZE, chunk_size)
    ]

    fd = os.open(temporary, os.O_CREAT | os.O_RDWR | os.O_TRUNC, 0o664)
    try:
        os.posix_fallocate(fd, 0, SIZE)
        completed = 0
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.workers
        ) as executor:
            futures = [
                executor.submit(download_range, fd, start, end)
                for start, end in ranges
            ]
            for future in concurrent.futures.as_completed(futures):
                completed += future.result()
                print(
                    f"SAM3_MODELSCOPE_PROGRESS {completed}/{SIZE} "
                    f"({completed / SIZE:.1%})",
                    flush=True,
                )
        os.fsync(fd)
    finally:
        os.close(fd)

    actual_sha256 = sha256(temporary)
    if actual_sha256 != SHA256:
        raise RuntimeError(
            f"SAM3 SHA256 mismatch: expected {SHA256}, got {actual_sha256}"
        )
    temporary.replace(args.output)
    print(f"SAM3_MODELSCOPE_DOWNLOAD_COMPLETE {args.output}", flush=True)


if __name__ == "__main__":
    main()

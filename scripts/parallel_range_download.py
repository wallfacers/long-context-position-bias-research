#!/usr/bin/env python3
"""Download one large HTTP artifact in verified, resumable byte ranges."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import math
import os
import shutil
import time
from pathlib import Path

import requests


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_range(
    url: str,
    path: Path,
    start: int,
    end: int,
    retries: int,
    timeout: float,
) -> tuple[int, int]:
    expected = end - start + 1
    path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, retries + 1):
        present = path.stat().st_size if path.exists() else 0
        if present == expected:
            return start, expected
        if present > expected:
            raise RuntimeError(f"Oversized partial file: {path}")
        request_start = start + present
        try:
            with requests.get(
                url,
                headers={"Range": f"bytes={request_start}-{end}"},
                stream=True,
                allow_redirects=True,
                timeout=(20, timeout),
            ) as response:
                response.raise_for_status()
                if response.status_code != 206:
                    raise RuntimeError(
                        f"Server ignored Range for {request_start}-{end}: "
                        f"HTTP {response.status_code}"
                    )
                content_range = response.headers.get("Content-Range", "")
                if not content_range.startswith(f"bytes {request_start}-{end}/"):
                    raise RuntimeError(f"Unexpected Content-Range: {content_range}")
                with path.open("ab") as handle:
                    for block in response.iter_content(chunk_size=1024 * 1024):
                        if block:
                            handle.write(block)
            if path.stat().st_size == expected:
                return start, expected
        except (OSError, requests.RequestException, RuntimeError) as exc:
            if attempt == retries:
                raise RuntimeError(
                    f"Range {start}-{end} failed after {retries} attempts"
                ) from exc
            time.sleep(min(2**attempt, 30))
    raise AssertionError("unreachable")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--parts", type=int, default=32)
    parser.add_argument("--retries", type=int, default=20)
    parser.add_argument("--timeout", type=float, default=120.0)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.size <= 0 or args.workers <= 0 or args.parts <= 0:
        raise SystemExit("size, workers, and parts must be positive")
    if args.output.exists():
        if args.output.stat().st_size == args.size and sha256_file(args.output) == args.sha256:
            print(f"Already verified: {args.output}")
            return 0
        raise SystemExit(f"Refusing to overwrite unverified output: {args.output}")

    parts_dir = args.output.with_name(args.output.name + ".parts")
    chunk_size = math.ceil(args.size / args.parts)
    ranges: list[tuple[int, int, Path]] = []
    for index in range(args.parts):
        start = index * chunk_size
        if start >= args.size:
            break
        end = min(start + chunk_size, args.size) - 1
        ranges.append((start, end, parts_dir / f"part-{index:04d}"))

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                download_range,
                args.url,
                path,
                start,
                end,
                args.retries,
                args.timeout,
            ): (start, end)
            for start, end, path in ranges
        }
        completed = 0
        for future in concurrent.futures.as_completed(futures):
            future.result()
            completed += 1
            print(f"ranges={completed}/{len(ranges)}", flush=True)

    temporary = args.output.with_name(args.output.name + f".tmp-{os.getpid()}")
    digest = hashlib.sha256()
    with temporary.open("wb") as destination:
        for _, _, path in ranges:
            with path.open("rb") as source:
                while block := source.read(4 * 1024 * 1024):
                    destination.write(block)
                    digest.update(block)
    if temporary.stat().st_size != args.size:
        raise SystemExit(f"Size mismatch: {temporary.stat().st_size} != {args.size}")
    actual_sha256 = digest.hexdigest()
    if actual_sha256 != args.sha256:
        raise SystemExit(f"SHA-256 mismatch: {actual_sha256} != {args.sha256}")
    temporary.replace(args.output)
    shutil.rmtree(parts_dir)
    print(f"Verified: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

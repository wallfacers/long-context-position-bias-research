#!/usr/bin/env python3
"""Fetch and deterministically reconstruct the frozen official NoLiMa sources.

The upstream ``download_NoLiMa_data.sh`` downloads ``rand_shuffle`` books and
then invokes ``wget -c`` on same-named ``rand_shuffle_long`` files.  With the
published files, that continuation keeps the normal-book prefix and appends
the long-book bytes after the normal file length.  The paper's frozen Qwen
data manifest was generated from those resulting bytes.  This script makes
that behavior explicit and verifies every raw and reconstructed SHA-256.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
import urllib.request
from pathlib import Path


DATASET_REVISION = "378115b1f136b6ba78f90f78682bc55f70ec3ddd"
OFFICIAL_REPOSITORY_REVISION = "cb14780b249fecf2851127b2101a062c1b2c6430"
BASE_URL = f"https://huggingface.co/datasets/amodaresi/NoLiMa/resolve/{DATASET_REVISION}"
NEEDLE_SHA256 = "8b4f17e3980503c2c2f6a72072d14df4caeec0d34d464d69c9ab73e1f1a8747c"
NEEDLE_BYTES = 4403
BOOKS = {
    1: {
        "normal_bytes": 418037,
        "normal": "1f7bdc697141b888565c851e6c34dbd649aaf580280eae92199561c4f0e93dab",
        "long_bytes": 1249102,
        "long": "fe77e62c40ef5d886848a16d79f0852da416c80c5e223267ab0ecf90f9fa1679",
        "combined": "290e84ffbb59b0a7af1a01b200d808692ee99018c220dace8b6d333cdac68cfe",
    },
    2: {
        "normal_bytes": 400808,
        "normal": "e77a1c8c52b31b0cb02b4b78af2e4f662ef7baaf0cfca52d84b33c710e353357",
        "long_bytes": 1265606,
        "long": "c67de75c02c355a21b625a90fbc25efa49952039e604595f31701a6e03e20ebb",
        "combined": "bc3a5245f0d556bb6d24e4064649c31267978529b58e6165d3a1d38191c33363",
    },
    3: {
        "normal_bytes": 420147,
        "normal": "3a45fdf98871e0a0e4924037661646b6ec20ecea490e2195fbd7103582782010",
        "long_bytes": 1246442,
        "long": "3d41dfa7941d0410370f2f34ba4e42fcad7fb711e2dfbf6751891df6a695a8c2",
        "combined": "6eac8fcb2bcd4bfc9c6298008463ba565c65485816a6f0eeb73cc7938ca7fa11",
    },
    4: {
        "normal_bytes": 414451,
        "normal": "9cf0ee84b8969c7a6a3b607d241af6b4f4dd4d6ab13ae8464d8cd4787ad5f618",
        "long_bytes": 1247457,
        "long": "51529ab20386878f0c7d7943e69cafa577654985d0b5dbecb518404896b37dda",
        "combined": "34cdad0fa7362ba270c4afc39f5acb3d65c39b00eefdf37b426cd0ac53f1a9b6",
    },
    5: {
        "normal_bytes": 390784,
        "normal": "0424d75836f692d04c07c14dad362828e89ff69c802c7021774a32ec289a62b9",
        "long_bytes": 1221569,
        "long": "02a97175dc2711a8b5a6f6b4a9129838787bf682dbc65588b5b762d68dbad817",
        "combined": "aeff17579398c5c995907d772b76783b93ec5f21ace4cf186f2f856e90aa10f8",
    },
}


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def reconstruct_continued_book(normal: bytes, long: bytes) -> bytes:
    if len(long) <= len(normal):
        raise ValueError("The frozen long book must be longer than the normal book")
    return normal + long[len(normal) :]


def download(url: str, expected_sha256: str, attempts: int = 3) -> bytes:
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            request = urllib.request.Request(
                url, headers={"User-Agent": "long-context-position-bias-repro/1"}
            )
            with urllib.request.urlopen(request, timeout=120) as response:
                payload = response.read()
            actual = sha256_bytes(payload)
            if actual != expected_sha256:
                raise ValueError(
                    f"Downloaded SHA-256 differs for {url}: {actual} != {expected_sha256}"
                )
            return payload
        except Exception as exc:  # network failures are retried, hash failures still fail closed
            last_error = exc
            if attempt < attempts:
                time.sleep(attempt)
    assert last_error is not None
    raise last_error


def refuse_mismatched_existing(path: Path, expected_sha256: str, overwrite: bool) -> None:
    if not path.exists():
        return
    if path.is_file() and sha256_bytes(path.read_bytes()) == expected_sha256:
        return
    if not overwrite:
        raise SystemExit(f"Existing output differs; use --overwrite explicitly: {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    needle_path = args.output_dir / "needlesets" / "needle_set_hard.json"
    book_dir = args.output_dir / "haystack" / "rand_shuffle"
    manifest_path = args.output_dir / "frozen-source-download-manifest.json"
    expected_outputs = {needle_path: NEEDLE_SHA256}
    expected_outputs.update(
        {book_dir / f"rand_book_{index}.txt": item["combined"] for index, item in BOOKS.items()}
    )
    for path, digest in expected_outputs.items():
        refuse_mismatched_existing(path, digest, args.overwrite)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs_complete = all(
        path.is_file() and sha256_bytes(path.read_bytes()) == digest
        for path, digest in expected_outputs.items()
    )
    raw_sources: dict[str, dict[str, object]] = {}
    if not outputs_complete:
        with tempfile.TemporaryDirectory(prefix="nolima-download-", dir=args.output_dir) as directory:
            temporary = Path(directory)
            needle = download(
                f"{BASE_URL}/needlesets/needle_set_hard.json", NEEDLE_SHA256
            )
            if len(needle) != NEEDLE_BYTES:
                raise SystemExit("Frozen NoLiMa needle byte size differs")
            staged: dict[Path, bytes] = {needle_path: needle}
            for index, expected in BOOKS.items():
                normal_url = f"{BASE_URL}/haystack/rand_shuffle/rand_book_{index}.txt"
                long_url = f"{BASE_URL}/haystack/rand_shuffle_long/rand_book_{index}.txt"
                normal = download(normal_url, expected["normal"])
                long = download(long_url, expected["long"])
                if len(normal) != expected["normal_bytes"] or len(long) != expected["long_bytes"]:
                    raise SystemExit(f"Frozen NoLiMa raw book size differs: {index}")
                combined = reconstruct_continued_book(normal, long)
                combined_sha = sha256_bytes(combined)
                if combined_sha != expected["combined"]:
                    raise SystemExit(
                        f"Reconstructed book {index} differs: {combined_sha} != {expected['combined']}"
                    )
                staged[book_dir / f"rand_book_{index}.txt"] = combined

            for index, (destination, payload) in enumerate(staged.items()):
                staged_path = temporary / f"{index}.part"
                staged_path.write_bytes(payload)
                destination.parent.mkdir(parents=True, exist_ok=True)
                os.replace(staged_path, destination)

    for index, expected in BOOKS.items():
        normal_url = f"{BASE_URL}/haystack/rand_shuffle/rand_book_{index}.txt"
        long_url = f"{BASE_URL}/haystack/rand_shuffle_long/rand_book_{index}.txt"
        combined_path = book_dir / f"rand_book_{index}.txt"
        if (
            not combined_path.is_file()
            or combined_path.stat().st_size != expected["long_bytes"]
            or sha256_bytes(combined_path.read_bytes()) != expected["combined"]
        ):
            raise SystemExit(f"Frozen NoLiMa final book validation failed: {index}")
        raw_sources[str(index)] = {
            "normal_url": normal_url,
            "normal_bytes": expected["normal_bytes"],
            "normal_sha256": expected["normal"],
            "long_url": long_url,
            "long_bytes": expected["long_bytes"],
            "long_sha256": expected["long"],
            "combined_bytes": combined_path.stat().st_size,
            "combined_sha256": expected["combined"],
        }

    manifest = {
        "schema_version": "nolima-frozen-source-download-v1",
        "status": "validated",
        "source": "Adobe Research NoLiMa official repository and dataset",
        "source_license": "Adobe Research License; noncommercial research only",
        "official_repository_revision": OFFICIAL_REPOSITORY_REVISION,
        "dataset_revision": DATASET_REVISION,
        "needle": {
            "url": f"{BASE_URL}/needlesets/needle_set_hard.json",
            "bytes": NEEDLE_BYTES,
            "sha256": NEEDLE_SHA256,
        },
        "books": raw_sources,
        "reconstruction": (
            "normal_bytes + long_bytes[len(normal_bytes):], explicitly reproducing "
            "the upstream download_NoLiMa_data.sh wget -c same-name continuation"
        ),
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Validated frozen NoLiMa sources at {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

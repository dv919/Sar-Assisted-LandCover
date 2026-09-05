"""
Download the two official BigEarthNet v2.0 metadata parquet files from Zenodo record 10891137.
Small (~4.3 MB total), safe, no imagery involved.
"""
import hashlib
import pathlib
import sys

import requests

ZENODO_RECORD = "10891137"
FILES = {
    "metadata.parquet": None,
    "metadata_for_patches_with_snow_cloud_or_shadow.parquet": None,
}
BASE_URL = f"https://zenodo.org/records/{ZENODO_RECORD}/files"
DEST = pathlib.Path(__file__).resolve().parent.parent / "data"
DEST.mkdir(parents=True, exist_ok=True)


def download(fname: str) -> pathlib.Path:
    out_path = DEST / fname
    if out_path.exists() and out_path.stat().st_size > 0:
        print(f"[skip] {fname} already present ({out_path.stat().st_size} bytes)")
        return out_path
    url = f"{BASE_URL}/{fname}?download=1"
    print(f"[get] {url}")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        written = 0
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                written += len(chunk)
        print(f"[done] {fname}: {written} bytes (expected {total})")
    return out_path


def main():
    for fname in FILES:
        download(fname)


if __name__ == "__main__":
    main()

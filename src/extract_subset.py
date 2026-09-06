"""
Stream-extract only the patches we need from the two official BigEarthNet v2.0 .tar.zst
archives on Zenodo, writing just the matching GeoTIFF band files to disk and aborting the
HTTP connection as soon as we've passed the last needed product (both archives are single,
non-seekable zstd frames -- confirmed by probe_archive.py -- so this is a sequential
read-and-filter, not a random-access fetch).

Archive layout (confirmed empirically):
  BigEarthNet-S2/<s2_product>/<patch_id>/<patch_id>_<BAND>.tif
  BigEarthNet-S1/<s1_product>/<s1_name>/<s1_name>_<VH|VV>.tif
where s2_product = patch_id minus trailing "_row_col", and
      s1_product = s1_name minus trailing "_tile_row_col" (one S1 scene spans many tiles).

Reads data/subset_patches.csv (written by select_subset.py) to know exactly which S2 patch_ids
and S1 s1_names to keep. Safe to re-run: already-written files are skipped.
"""
import pathlib
import time

import pandas as pd
import requests
import tarfile
import zstandard as zstd

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
ZENODO_RECORD = "10891137"
MAX_RETRIES = 8
RETRY_BACKOFF_S = 15


def s2_product_name(patch_id: str) -> str:
    return "_".join(patch_id.split("_")[:-2])


def s1_product_name(s1_name: str) -> str:
    return "_".join(s1_name.split("_")[:-3])


def _already_complete(item_id: str, out_root: pathlib.Path, expected_suffixes) -> bool:
    d = out_root / item_id
    if not d.exists():
        return False
    have = {p.name.rsplit("_", 1)[-1].split(".")[0] for p in d.glob("*.tif")}
    return expected_suffixes.issubset(have)


def stream_extract_once(fname: str, remaining_ids: set, last_needed_product: str,
                         out_root: pathlib.Path, log_every: int = 500):
    """Single attempt: stream from byte 0, write matching files, abort once past the last
    needed product OR once every remaining id has been found. Raises on connection failure
    (caller retries); files already on disk are skipped so retries are cheap on the disk side."""
    url = f"https://zenodo.org/records/{ZENODO_RECORD}/files/{fname}?download=1"
    t0 = time.time()
    r = requests.get(url, stream=True, timeout=(15, 90))
    r.raise_for_status()
    raw = r.raw
    raw.decode_content = True
    dctx = zstd.ZstdDecompressor()
    reader = dctx.stream_reader(raw, read_across_frames=False)
    tf = tarfile.open(fileobj=reader, mode="r|")

    n_members = 0
    n_written = 0
    n_bytes_written = 0
    seen_products = set()
    found_ids = set()
    still_needed = set(remaining_ids)
    try:
        for member in tf:
            n_members += 1
            parts = member.name.split("/")
            if len(parts) < 4:
                continue  # a directory entry, not a band file
            product, item_id, filename = parts[1], parts[2], parts[3]
            if product not in seen_products:
                seen_products.add(product)
                if product > last_needed_product:
                    print(f"[{fname}] reached product {product} > last needed {last_needed_product}; stopping.")
                    break

            if item_id in still_needed:
                found_ids.add(item_id)
                out_dir = out_root / item_id
                out_path = out_dir / filename
                if not out_path.exists():
                    out_dir.mkdir(parents=True, exist_ok=True)
                    data = tf.extractfile(member).read()
                    out_path.write_bytes(data)
                    n_written += 1
                    n_bytes_written += len(data)

            if n_members % log_every == 0:
                elapsed = time.time() - t0
                print(f"[{fname}] members={n_members} products_seen={len(seen_products)} "
                      f"ids_found_this_pass={len(found_ids)}/{len(still_needed)} written={n_written} "
                      f"({n_bytes_written/1e6:.1f} MB) elapsed={elapsed:.0f}s", flush=True)
    finally:
        r.close()

    print(f"[{fname}] pass done: members={n_members} written={n_written} "
          f"bytes={n_bytes_written/1e6:.1f}MB elapsed={time.time()-t0:.0f}s")
    return found_ids


def stream_extract(fname: str, needed_ids: set, last_needed_product: str,
                    out_root: pathlib.Path, expected_suffixes: set):
    out_root.mkdir(parents=True, exist_ok=True)

    remaining = {i for i in needed_ids if not _already_complete(i, out_root, expected_suffixes)}
    print(f"[{fname}] {len(needed_ids)} needed ids total; {len(needed_ids) - len(remaining)} "
          f"already complete on disk; {len(remaining)} remaining.")

    attempt = 0
    while remaining and attempt < MAX_RETRIES:
        attempt += 1
        print(f"[{fname}] attempt {attempt}/{MAX_RETRIES}, {len(remaining)} ids remaining ...")
        try:
            stream_extract_once(fname, remaining, last_needed_product, out_root)
        except Exception as e:
            print(f"[{fname}] attempt {attempt} failed: {type(e).__name__}: {e}")
        remaining = {i for i in needed_ids if not _already_complete(i, out_root, expected_suffixes)}
        if remaining and attempt < MAX_RETRIES:
            print(f"[{fname}] {len(remaining)} ids still incomplete; retrying in {RETRY_BACKOFF_S}s ...")
            time.sleep(RETRY_BACKOFF_S)

    found_ids = needed_ids - remaining
    print(f"[{fname}] FINAL: found={len(found_ids)}/{len(needed_ids)} missing={len(remaining)}")
    if remaining:
        print(f"[{fname}] WARNING still missing after {attempt} attempts (up to 10 shown): "
              f"{list(remaining)[:10]}")
    return found_ids, remaining


def main():
    import sys
    subset_csv = sys.argv[1] if len(sys.argv) > 1 else "subset_patches.csv"
    print(f"Using subset file: {subset_csv}")
    subset = pd.read_csv(DATA_DIR / subset_csv)
    s2_needed = set(subset["patch_id"])
    s1_needed = set(subset["s1_name"])

    last_s2_product = max(s2_product_name(p) for p in s2_needed)
    last_s1_product = max(s1_product_name(p) for p in s1_needed)
    print(f"S2 needed patches: {len(s2_needed)}; last needed product: {last_s2_product}")
    print(f"S1 needed scenes: {len(s1_needed)}; last needed product: {last_s1_product}")

    s2_bands = {"B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"}
    s1_bands = {"VV", "VH"}

    found_s2, missing_s2 = stream_extract(
        "BigEarthNet-S2.tar.zst", s2_needed, last_s2_product, RAW_DIR / "S2", s2_bands
    )
    found_s1, missing_s1 = stream_extract(
        "BigEarthNet-S1.tar.zst", s1_needed, last_s1_product, RAW_DIR / "S1", s1_bands
    )

    print("SUMMARY:", {
        "s2_found": len(found_s2), "s2_missing": len(missing_s2),
        "s1_found": len(found_s1), "s1_missing": len(missing_s1),
    })


if __name__ == "__main__":
    main()

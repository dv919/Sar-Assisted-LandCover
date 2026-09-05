"""
Cheap, bounded probe: read the first N MB of BigEarthNet-S2.tar.zst and
BigEarthNet-S1.tar.zst via HTTP range request, stream-decompress, and list the
first several tar member names so extract_subset.py can match real paths
instead of guessed ones. No full download.
"""
import io
import sys

import requests
import tarfile
import zstandard as zstd

ZENODO_RECORD = "10891137"
RANGE_BYTES = 30 * 1024 * 1024  # 30 MB, same order as the earlier Phase-1 probe


def probe(fname: str, max_members: int = 60):
    url = f"https://zenodo.org/records/{ZENODO_RECORD}/files/{fname}?download=1"
    headers = {"Range": f"bytes=0-{RANGE_BYTES - 1}"}
    print(f"\n=== {fname} ===")
    r = requests.get(url, headers=headers, stream=True, timeout=60)
    print("status:", r.status_code, "content-range:", r.headers.get("content-range"))
    raw = r.raw
    raw.decode_content = True
    dctx = zstd.ZstdDecompressor()
    reader = dctx.stream_reader(raw, read_across_frames=False)
    tf = tarfile.open(fileobj=reader, mode="r|")
    count = 0
    try:
        for member in tf:
            print(member.name, member.size, member.type)
            count += 1
            if count >= max_members:
                break
    except Exception as e:
        print("stopped iterating:", type(e).__name__, e)
    finally:
        r.close()


if __name__ == "__main__":
    probe("BigEarthNet-S2.tar.zst")
    probe("BigEarthNet-S1.tar.zst")

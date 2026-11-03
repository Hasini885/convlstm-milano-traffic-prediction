"""Download the Telecom Italia Milano daily files from Harvard Dataverse.

Dataset: "Telecommunications - SMS, Call, Internet - MI", doi:10.7910/DVN/EGZHFV
(62 public daily files, about 20.8 GB uncompressed). No account is needed.

With ``--gzip`` (the default) each file is compressed while it downloads, so it needs about a
fifth of the disk space. ``src.dataset`` reads ``.txt.gz`` files directly. Files that are already
complete are skipped, so an interrupted download can simply be restarted.

Usage::

    python scripts/download_dataverse.py --out data/raw [--max-days 7] [--no-gzip]
"""
from __future__ import annotations

import argparse
import gzip
import json
import shutil
import sys
import time
import urllib.request
from pathlib import Path

DATAVERSE = "https://dataverse.harvard.edu"
DOI = "doi:10.7910/DVN/EGZHFV"
PATTERN = "sms-call-internet-mi-"
CHUNK = 8 * 1024 * 1024
# The server rejects urllib's default User-Agent with 403.
HEADERS = {"User-Agent": "Mozilla/5.0 (convlstm-milano downloader)"}


def _open(url: str, timeout: int):
    return urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=timeout)


def list_files(doi: str = DOI) -> list[dict]:
    """Return ``[{"id", "filename", "size"}, ...]`` for the daily files, sorted by name."""
    url = f"{DATAVERSE}/api/datasets/:persistentId/?persistentId={doi}"
    with _open(url, 120) as resp:
        data = json.load(resp)
    files = [
        {"id": f["dataFile"]["id"], "filename": f["dataFile"]["filename"], "size": f["dataFile"].get("filesize")}
        for f in data["data"]["latestVersion"]["files"]
        if f["dataFile"]["filename"].startswith(PATTERN)
    ]
    return sorted(files, key=lambda f: f["filename"])


def download(file: dict, out_dir: Path, use_gzip: bool, retries: int = 5) -> Path:
    """Download one file (optionally gzip-compressing on the fly), with retries."""
    target = out_dir / (file["filename"] + (".gz" if use_gzip else ""))
    marker = target.with_name(target.name + ".done")
    if target.exists() and marker.exists():
        return target
    url = f"{DATAVERSE}/api/access/datafile/{file['id']}"
    tmp = target.with_name(target.name + ".part")
    for attempt in range(1, retries + 1):
        try:
            received = 0
            with _open(url, 300) as resp:
                sink = gzip.open(tmp, "wb", compresslevel=1) if use_gzip else open(tmp, "wb")
                with sink:
                    while chunk := resp.read(CHUNK):
                        sink.write(chunk)
                        received += len(chunk)
            if file["size"] and received != file["size"]:
                raise IOError(f"size mismatch: got {received}, expected {file['size']}")
            tmp.replace(target)
            marker.touch()
            return target
        except Exception as exc:  # network errors: retry with backoff
            print(f"  attempt {attempt}/{retries} for {file['filename']} failed: {exc}", flush=True)
            tmp.unlink(missing_ok=True)
            time.sleep(min(60, 5 * attempt))
    raise RuntimeError(f"Could not download {file['filename']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="data/raw", help="Output directory")
    parser.add_argument("--max-days", type=int, default=None, help="Only the first N days")
    parser.add_argument("--no-gzip", action="store_true", help="Store plain .txt files")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    files = list_files()[: args.max_days]
    total = sum(f["size"] or 0 for f in files)
    print(f"{len(files)} files, {total / 1e9:.1f} GB uncompressed -> {out} "
          f"(free disk: {shutil.disk_usage(out).free / 1e9:.1f} GB)", flush=True)
    start = time.time()
    for i, f in enumerate(files, 1):
        path = download(f, out, not args.no_gzip)
        print(f"  [{i}/{len(files)}] {path.name}  ({path.stat().st_size / 1e6:.0f} MB on disk, "
              f"{time.time() - start:.0f}s elapsed)", flush=True)
    print("Download complete.", flush=True)


if __name__ == "__main__":
    sys.exit(main())

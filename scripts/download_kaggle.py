"""Download the Telecom Italia Milano daily files from the Kaggle mirror.

Dataset: ``dkgmgo/telecom-italia-milan`` (the 62 daily files of the Harvard Dataverse release
doi:10.7910/DVN/EGZHFV, about 20.8 GB uncompressed, plus ``milano-grid.geojson``).

The Dataverse itself cannot be scripted: its API answers every file request with
"You may not download this file without the required Guestbook response", so files from there
have to be fetched by hand in a browser.

Needs the Kaggle CLI (``pip install kaggle``) and an API token (https://www.kaggle.com/settings).
Files that already exist with the right size are skipped, so an interrupted download can simply be
restarted.

Usage::

    python scripts/download_kaggle.py --out data/raw [--max-days 7]
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import zipfile
from datetime import date, timedelta
from pathlib import Path

DATASET = "dkgmgo/telecom-italia-milan"
FIRST_DAY, NUM_DAYS = date(2013, 11, 1), 62


def download(name: str, out_dir: Path) -> Path:
    """Download ``data/<name>`` from the dataset into ``out_dir`` and return its path."""
    target = out_dir / name
    if target.exists() and target.stat().st_size > 0:
        return target
    subprocess.run(["kaggle", "datasets", "download", DATASET, "-f", f"data/{name}", "-p", str(out_dir), "-q"],
                   check=True)
    # Depending on the CLI version, the file lands under data/ and/or as a .zip; normalise both.
    for candidate in (out_dir / "data" / name, out_dir / f"{name}.zip"):
        if candidate.suffix == ".zip" and candidate.exists():
            with zipfile.ZipFile(candidate) as z:
                z.extractall(out_dir)
            candidate.unlink()
        elif candidate.exists():
            candidate.replace(target)
    if not target.exists():
        raise FileNotFoundError(f"kaggle reported success but {target} is missing")
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="data/raw", help="Output directory")
    parser.add_argument("--max-days", type=int, default=NUM_DAYS, help="Only the first N days")
    args = parser.parse_args()

    if shutil.which("kaggle") is None:
        sys.exit("The Kaggle CLI is not installed. Run `pip install kaggle` and add an API token.")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    days = [FIRST_DAY + timedelta(days=i) for i in range(min(args.max_days, NUM_DAYS))]
    print(f"{len(days)} daily files (about 0.33 GB each) -> {out} "
          f"(free disk: {shutil.disk_usage(out).free / 1e9:.1f} GB)", flush=True)
    for i, day in enumerate(days, 1):
        path = download(f"sms-call-internet-mi-{day.isoformat()}.txt", out)
        print(f"  [{i}/{len(days)}] {path.name}  ({path.stat().st_size / 1e6:.0f} MB)", flush=True)
    leftover = out / "data"
    if leftover.is_dir() and not any(leftover.iterdir()):
        leftover.rmdir()
    print("Download complete.", flush=True)


if __name__ == "__main__":
    main()

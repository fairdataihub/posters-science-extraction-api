"""
Upload poster thumbnails to Bunny public storage.

Source folders:
  /Users/sanjay/Downloads/thumbnails/figshare
  /Users/sanjay/Downloads/thumbnails/zenodo

Destination path on Bunny: thumbnails/a/<filename>
  e.g. figshare_10005125.jpeg → thumbnails/a/figshare_10005125.jpeg

Reads BUNNY_PUBLIC_STORAGE and BUNNY_PUBLIC_STORAGE_KEY from .env
"""

import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from dotenv import dotenv_values
from tqdm import tqdm

ENV = dotenv_values(Path(__file__).parent / ".env")

BUNNY_BASE_URL = ENV.get("BUNNY_PUBLIC_STORAGE", "").rstrip("/")
BUNNY_KEY = ENV.get("BUNNY_PUBLIC_STORAGE_KEY", "")

if not BUNNY_BASE_URL or not BUNNY_KEY:
    print("ERROR: BUNNY_PUBLIC_STORAGE or BUNNY_PUBLIC_STORAGE_KEY not set in .env", file=sys.stderr)
    sys.exit(1)

SOURCE_DIRS = [
    Path("/Users/sanjay/Downloads/thumbnails/figshare"),
    Path("/Users/sanjay/Downloads/thumbnails/zenodo"),
]

HEADERS = {
    "AccessKey": BUNNY_KEY,
    "Content-Type": "application/octet-stream",
}


def upload_file(local_path: Path) -> None:
    url = f"{BUNNY_BASE_URL}/thumbnails/a/{local_path.name}"
    with local_path.open("rb") as f:
        resp = requests.put(url, data=f, headers=HEADERS, timeout=60)
    if resp.status_code not in (200, 201):
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")


def main() -> None:
    files: list[Path] = []
    for src_dir in SOURCE_DIRS:
        if not src_dir.exists():
            print(f"WARNING: {src_dir} not found, skipping", file=sys.stderr)
            continue
        found = sorted(p for p in src_dir.glob("*.jpeg") if not p.name.startswith("._"))
        print(f"[{src_dir.name}] {len(found)} files")
        files.extend(found)

    print(f"\nTotal: {len(files)} files")
    if not files:
        return

    errors: list[tuple[str, str]] = []

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(upload_file, p): p for p in files}
        with tqdm(total=len(futures), desc="Uploading", unit="file") as pbar:
            for future in as_completed(futures):
                local_path = futures[future]
                try:
                    future.result()
                except Exception as exc:
                    errors.append((local_path.name, str(exc)))
                    tqdm.write(f"  ERR {local_path.name}: {exc}", file=sys.stderr)
                finally:
                    pbar.update(1)

    print(f"\nDone. {len(errors)} error(s).")
    if errors:
        for name, msg in errors:
            print(f"  {name}: {msg}")
        sys.exit(1)


if __name__ == "__main__":
    main()

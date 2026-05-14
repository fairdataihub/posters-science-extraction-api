"""
Generate thumbnails from auto-extracted poster PDFs stored on the external drive.

Input folders:
  /Volumes/Crucial X10/posters/posters/figshare
  /Volumes/Crucial X10/posters/posters/zenodo

Filename convention: <source>_<id>_<anything>.pdf
  e.g. figshare_5354356_FinalMidsurePoster.pdf → thumbnail saved as figshare_5354356.jpeg

Output folders:
  /Volumes/Crucial X10/posters/thumbnails/figshare
  /Volumes/Crucial X10/posters/thumbnails/zenodo

Thumbnail settings:
  MAX_WIDTH    = 1200 px
  JPEG_QUALITY = 75
"""

import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import os

import pymupdf  # PyMuPDF
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
MAX_WIDTH = 1200
JPEG_QUALITY = 75
MAX_WORKERS = min(16, max(1, os.cpu_count() or 1))

DRIVE = Path("/Volumes/Crucial X10/posters")

SOURCES = ["figshare", "zenodo"]


# ---------------------------------------------------------------------------
# Core conversion
# ---------------------------------------------------------------------------
def convert_pdf_to_thumbnail(pdf_path_str: str, output_path_str: str) -> None:
    pdf_path = Path(pdf_path_str)
    output_path = Path(output_path_str)

    doc = pymupdf.open(pdf_path)
    try:
        page = doc[0]
        scale = min(1.0, MAX_WIDTH / page.rect.width)
        pix = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pix.save(str(output_path), jpg_quality=JPEG_QUALITY)
    finally:
        doc.close()


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------
def main() -> None:
    tasks: list[tuple[Path, Path]] = []
    skipped = 0

    for source in SOURCES:
        src_dir = DRIVE / "posters" / source
        dst_dir = DRIVE / "thumbnails" / source

        dst_dir.mkdir(parents=True, exist_ok=True)

        pdf_files = sorted(p for p in src_dir.glob("*.pdf") if not p.name.startswith("._"))
        if not pdf_files:
            print(f"[{source}] No PDFs found — skipping.")
            continue

        for pdf_path in pdf_files:
            parts = pdf_path.stem.split("_")
            if len(parts) < 2:
                print(f"[{source}] Unexpected filename, skipping: {pdf_path.name}")
                continue
            poster_id = parts[1]
            output_path = dst_dir / f"{source}_{poster_id}.jpeg"

            if output_path.exists():
                skipped += 1
                continue

            tasks.append((pdf_path, output_path))

        print(f"[{source}] {len(pdf_files)} PDFs — {skipped} already done, queuing remainder.")

    print(f"\nTotal to convert: {len(tasks)}  (skipped existing: {skipped})")
    if not tasks:
        print("Nothing to do.")
        return

    print(f"Using up to {MAX_WORKERS} parallel workers (processes)...\n")

    errors: list[tuple[Path, str]] = []
    start = time.perf_counter()

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(convert_pdf_to_thumbnail, str(pdf_path), str(output_path)): pdf_path
            for pdf_path, output_path in tasks
        }
        with tqdm(total=len(futures), desc="Thumbnailing", unit="pdf") as pbar:
            for future in as_completed(futures):
                pdf_path = futures[future]
                try:
                    future.result()
                except Exception as exc:
                    errors.append((pdf_path, str(exc)))
                    tqdm.write(f"  ERR {pdf_path.name}: {exc}", file=sys.stderr)
                finally:
                    pbar.update(1)

    elapsed = time.perf_counter() - start
    rate = len(tasks) / elapsed if elapsed > 0 else 0.0

    print(f"\nDone. {len(errors)} error(s).")
    print(f"Elapsed: {elapsed:.1f}s  Rate: {rate:.2f} pdf/s")
    if errors:
        print("Failed files:")
        for path, msg in errors:
            print(f"  {path.name}: {msg}")
        sys.exit(1)


if __name__ == "__main__":
    main()

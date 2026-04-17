"""
Convert all PDFs in poster_sentry_review_sample_350 (poster/ and non_poster/)
to JPEG images, preserving the subfolder structure under an output directory.

Quality settings — higher than thumbnail:
  MAX_WIDTH    = 2400 px   (2x the thumbnail 1200)
  JPEG_QUALITY = 90        (vs thumbnail's 75)
"""

import sys
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pymupdf  # PyMuPDF
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
MAX_WIDTH = 2400  # px — 2x thumbnail width for higher fidelity
JPEG_QUALITY = 90  # 1-100; noticeably sharper than thumbnail quality 75
MAX_WORKERS = 8

INPUT_ROOT = Path(__file__).parent / "poster_sentry_training_pdfs"
OUTPUT_ROOT = Path(__file__).parent / "poster_sentry_training_pdfs_jpeg"

SUBFOLDERS = ["poster", "non_poster"]


# ---------------------------------------------------------------------------
# Core conversion
# ---------------------------------------------------------------------------
def convert_pdf_to_jpeg(pdf_path: Path, output_path: Path, page_num: int = 0) -> None:
    """Convert a single PDF page to a high-quality JPEG."""
    doc = pymupdf.open(pdf_path)
    try:
        page = doc[page_num]
        natural_width = page.rect.width  # points at 72 dpi
        scale = min(1.0, MAX_WIDTH / natural_width)
        matrix = pymupdf.Matrix(scale, scale)
        pix = page.get_pixmap(matrix=matrix)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pix.save(str(output_path), jpg_quality=JPEG_QUALITY)
        print(
            f"  OK  {output_path.relative_to(OUTPUT_ROOT)}  "
            f"({pix.width}x{pix.height}px, scale={scale:.2f})"
        )
    finally:
        doc.close()


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------
def main() -> None:
    errors: list[tuple[Path, str]] = []

    tasks: list[tuple[Path, Path]] = []

    # Delete this script's output directory if it already exists, to ensure a clean slate.
    if OUTPUT_ROOT.exists():
        os.system(f"rm -rf {OUTPUT_ROOT}")
        print(f"Deleted existing output directory: {OUTPUT_ROOT}")

    for subfolder in SUBFOLDERS:
        src_dir = INPUT_ROOT / subfolder
        dst_dir = OUTPUT_ROOT / subfolder

        pdf_files = sorted(src_dir.glob("*.pdf"))
        if not pdf_files:
            print(f"[{subfolder}] No PDFs found — skipping.")
            continue

        print(f"\n[{subfolder}] Queuing {len(pdf_files)} PDFs → {dst_dir}")
        tasks.extend((pdf_path, dst_dir / f"{pdf_path.stem}.jpg") for pdf_path in pdf_files)

    print(f"\nTotal PDFs to convert: {len(tasks)}")
    print(f"Using up to {MAX_WORKERS} parallel workers...")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(convert_pdf_to_jpeg, pdf_path, output_path): pdf_path
            for pdf_path, output_path in tasks
        }
        with tqdm(total=len(futures), desc="Converting", unit="pdf") as pbar:
            for future in as_completed(futures):
                pdf_path = futures[future]
                try:
                    future.result()
                except Exception as exc:
                    print(f"  ERR {pdf_path.name}: {exc}", file=sys.stderr)
                    errors.append((pdf_path, str(exc)))
                finally:
                    pbar.update(1)

    print(f"\nDone. {len(errors)} error(s).")
    if errors:
        print("Failed files:")
        for path, msg in errors:
            print(f"  {path.name}: {msg}")
        sys.exit(1)


if __name__ == "__main__":
    main()

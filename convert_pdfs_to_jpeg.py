"""
Convert all PDFs in poster_sentry_review_sample_350 (poster/ and non_poster/)
to JPEG images, preserving the subfolder structure under an output directory.

Quality settings — higher than thumbnail:
  MAX_WIDTH    = 2400 px   (2x the thumbnail 1200)
  JPEG_QUALITY = 90        (vs thumbnail's 75)
"""

import sys
from pathlib import Path

import pymupdf  # PyMuPDF

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
MAX_WIDTH = 2400       # px — 2x thumbnail width for higher fidelity
JPEG_QUALITY = 90      # 1-100; noticeably sharper than thumbnail quality 75

INPUT_ROOT = Path(__file__).parent / "poster_sentry_review_sample_350"
OUTPUT_ROOT = Path(__file__).parent / "poster_sentry_review_sample_350_jpeg"

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
        print(f"  OK  {output_path.relative_to(OUTPUT_ROOT)}  "
              f"({pix.width}x{pix.height}px, scale={scale:.2f})")
    finally:
        doc.close()


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------
def main() -> None:
    errors: list[tuple[Path, str]] = []

    for subfolder in SUBFOLDERS:
        src_dir = INPUT_ROOT / subfolder
        dst_dir = OUTPUT_ROOT / subfolder

        pdf_files = sorted(src_dir.glob("*.pdf"))
        if not pdf_files:
            print(f"[{subfolder}] No PDFs found — skipping.")
            continue

        print(f"\n[{subfolder}] Converting {len(pdf_files)} PDFs → {dst_dir}")

        for pdf_path in pdf_files:
            output_path = dst_dir / (pdf_path.stem + ".jpg")
            try:
                convert_pdf_to_jpeg(pdf_path, output_path)
            except Exception as exc:
                print(f"  ERR {pdf_path.name}: {exc}", file=sys.stderr)
                errors.append((pdf_path, str(exc)))

    print(f"\nDone. {len(errors)} error(s).")
    if errors:
        print("Failed files:")
        for path, msg in errors:
            print(f"  {path.name}: {msg}")
        sys.exit(1)


if __name__ == "__main__":
    main()

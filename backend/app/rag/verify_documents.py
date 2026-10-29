from pathlib import Path
from pypdf import PdfReader

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PDF_ROOT = PROJECT_ROOT / "data" / "rag_documents"

print(f"Project root: {PROJECT_ROOT}")
print(f"PDF directory: {PDF_ROOT}")
print("-" * 80)

if not PDF_ROOT.exists():
    print(f"ERROR: PDF directory does not exist: {PDF_ROOT}")
    raise SystemExit(1)

pdf_files = sorted(
    path for path in PDF_ROOT.rglob("*")
    if path.is_file() and path.suffix.lower() == ".pdf"
)

if not pdf_files:
    print("No PDF files found. Check the directory path.")
    raise SystemExit(1)

valid_count = 0
needs_review_count = 0
invalid_count = 0

for pdf_path in pdf_files:
    relative_path = pdf_path.relative_to(PDF_ROOT)

    try:
        reader = PdfReader(str(pdf_path), strict=False)

        if reader.is_encrypted:
            print(f"[LOCKED]  {relative_path} — encrypted PDF")
            needs_review_count += 1
            continue

        page_count = len(reader.pages)

        if page_count == 0:
            print(f"[INVALID] {relative_path} — no pages")
            invalid_count += 1
            continue

        page_texts = []

        for page in reader.pages:
            try:
                page_texts.append(page.extract_text() or "")
            except Exception:
                page_texts.append("")

        extracted_text = "\n".join(page_texts).strip()
        character_count = len(extracted_text)

        if character_count < 100:
            print(
                f"[CHECK]   {relative_path} — "
                f"{page_count} pages, "
                f"{character_count} extracted characters"
            )
            print(
                "          May be scanned, image-only, "
                "empty, or incorrectly downloaded."
            )
            needs_review_count += 1

        else:
            print(
                f"[OK]      {relative_path} — "
                f"{page_count} pages, "
                f"{character_count:,} extracted characters"
            )
            valid_count += 1

    except Exception as exc:
        print(f"[INVALID] {relative_path} — {exc}")
        invalid_count += 1

print("-" * 80)
print(f"Readable PDFs: {valid_count}")
print(f"Files requiring inspection: {needs_review_count}")
print(f"Files that could not be read: {invalid_count}")
print(f"Total PDFs: {len(pdf_files)}")
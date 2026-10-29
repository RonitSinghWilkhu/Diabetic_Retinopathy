from pathlib import Path
from pypdf import PdfReader

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PDF_ROOT = PROJECT_ROOT / "data" / "rag_documents"

pdf_files = sorted(
    path for path in PDF_ROOT.rglob("*")
    if path.is_file() and path.suffix.lower() == ".pdf"
)

for pdf_path in pdf_files:
    relative_path = pdf_path.relative_to(PDF_ROOT)

    print("\n" + "=" * 90)
    print(f"FILE: {relative_path}")

    try:
        reader = PdfReader(str(pdf_path), strict=False)

        metadata = reader.metadata

        if metadata:
            print(f"PDF title: {metadata.title}")
            print(f"PDF author: {metadata.author}")
            print(f"PDF creator: {metadata.creator}")

        if len(reader.pages) > 0:
            first_page = reader.pages[0].extract_text() or ""
            print("\nFIRST PAGE:")
            print(first_page[:2500])

    except Exception as exc:
        print(f"ERROR: {exc}")

print("\n" + "=" * 90)
print("Document inspection complete.")
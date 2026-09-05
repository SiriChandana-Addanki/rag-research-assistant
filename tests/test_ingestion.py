from pathlib import Path

from src.ingestion import extract_text_from_pdf


def test_pdf_ingestion():
    pdf_path = Path("data/raw").glob("*.pdf")
    pdf_path = next(pdf_path, None)

    assert pdf_path is not None, "No PDF found in data/raw"

    pages = extract_text_from_pdf(pdf_path)

    assert len(pages) > 0
    assert "text" in pages[0]
    assert "metadata" in pages[0]
    assert "source" in pages[0]["metadata"]
    assert "page" in pages[0]["metadata"]
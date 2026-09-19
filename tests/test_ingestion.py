import pymupdf

from src.ingestion import extract_text_from_pdf


def test_pdf_ingestion(tmp_path):
    pdf_path = tmp_path / "sample.pdf"
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 72), "A short test document.")
    document.save(pdf_path)
    document.close()

    pages = extract_text_from_pdf(pdf_path)

    assert pages == [
        {
            "text": "A short test document.",
            "metadata": {"source": "sample.pdf", "page": 1},
        }
    ]

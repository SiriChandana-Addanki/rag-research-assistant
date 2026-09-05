from pathlib import Path
import fitz


def extract_text_from_pdf(pdf_path: Path) -> list[dict]:
    """
    Extract non - empty text from a PDF page by page.

    What does it Return?
        A list of dictionaries containing page text and metadata.
    """

    pages = []

    with fitz.open(pdf_path) as document:
        for page_number, page in enumerate(document, start=1):
            text = page.get_text("text").strip()

            if not text:
                continue

            pages.append(
            {
                "text": text,
                "metadata": {
                    "source": pdf_path.name,
                    "page": page_number,
                },
            }
        )

    

    return pages
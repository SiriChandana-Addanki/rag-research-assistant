from pathlib import Path

import pymupdf


def clean_text(text: str) -> str:
    """Clean common PDF extraction artifacts."""

    # Join words broken by PDF line wrapping.
    text = text.replace("-\n", "")

    # Convert remaining line breaks into spaces.
    text = text.replace("\n", " ")

    # Remove excessive whitespace.
    text = " ".join(text.split())

    return text.strip()


def extract_text_from_pdf(pdf_path: Path) -> list[dict]:
    """
    Extract and clean text from a PDF page by page.

    Returns:
        A list of page-level documents containing text and metadata.
    """

    document = pymupdf.open(pdf_path)

    pages = []

    for page_number, page in enumerate(document, start=1):
        raw_text = page.get_text("text")
        text = clean_text(raw_text)

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

    document.close()

    return pages
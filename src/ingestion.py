from pathlib import Path
import re

import pymupdf


def clean_text(text: str) -> str:
    """
    Clean PDF extraction artifacts while preserving meaningful line breaks.
    """

    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Join words broken across PDF line wrapping.
    text = re.sub(r"-\n(?=\w)", "", text)

    # Remove trailing whitespace from every extracted line.
    lines = [line.strip() for line in text.split("\n")]

    # Remove empty lines at the beginning and end.
    while lines and not lines[0]:
        lines.pop(0)

    while lines and not lines[-1]:
        lines.pop()

    return "\n".join(lines)


def clean_page_artifacts(text: str) -> str:
    """
    Remove common academic PDF artifacts such as page numbers
    and arXiv headers and footers.
    """

    cleaned_lines = []

    for line in text.split("\n"):
        line = line.strip()

        if not line:
            continue

        # arXiv identifier/header/footer.
        if re.match(r"^arXiv:\d+\.\d+", line):
            continue

        # Standalone page numbers.
        if re.fullmatch(r"\d+", line):
            continue

        # Footnote markers such as:
        # 1Our code...
        if re.match(r"^\d+[A-Z]", line):
            continue

        cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


def extract_text_blocks(page) -> list[dict]:
    """
    Extract text blocks while preserving their page coordinates.
    """

    blocks = page.get_text("blocks")

    text_blocks = []

    for block in blocks:

        x0, y0, x1, y1, text, block_number, block_type = block

        # Keep only text blocks.
        if block_type != 0:
            continue

        text = text.strip()

        if not text:
            continue

        text_blocks.append(
            {
                "text": text,
                "x0": x0,
                "y0": y0,
                "x1": x1,
                "y1": y1,
                "block_number": block_number,
            }
        )

    return text_blocks


def is_figure_caption(block: dict) -> bool:
    """
    Return True when a text block looks like a figure caption.
    """

    return bool(
        re.match(
            r"^(Figure|Fig\.)\s+\d+[:.]",
            block["text"].strip(),
            re.IGNORECASE,
        )
    )


def remove_figure_blocks(
    blocks: list[dict],
    page_width: float,
) -> list[dict]:
    """
    Remove text blocks belonging to figures while preserving
    the figure caption itself.

    Figure blocks are identified using the caption's position
    and horizontal extent rather than paper-specific text.
    """

    captions = [
        block
        for block in blocks
        if is_figure_caption(block)
    ]

    if not captions:
        return blocks

    filtered_blocks = []

    for block in blocks:

        remove_block = False

        for caption in captions:

            # Only consider blocks above the caption.
            if block["y1"] > caption["y0"]:
                continue

            # Calculate horizontal overlap.
            overlap_start = max(
                block["x0"],
                caption["x0"],
            )

            overlap_end = min(
                block["x1"],
                caption["x1"],
            )

            overlap_width = max(
                0,
                overlap_end - overlap_start,
            )

            block_width = block["x1"] - block["x0"]

            if block_width == 0:
                continue

            overlap_ratio = overlap_width / block_width

            # Figure blocks tend to occupy a region that overlaps
            # strongly with the caption horizontally.
            if overlap_ratio >= 0.8:

                # Avoid removing normal full-width body text.
                is_full_width = (
                    block["x0"] <= page_width * 0.15
                    and block["x1"] >= page_width * 0.85
                )

                if not is_full_width:
                    remove_block = True
                    break

        if not remove_block:
            filtered_blocks.append(block)

    return filtered_blocks


def extract_text_from_pdf(pdf_path: Path) -> list[dict]:
    """
    Extract text from a PDF page by page using layout-aware
    text blocks.

    Returns:
        A list of page-level documents containing text and metadata.
    """

    document = pymupdf.open(pdf_path)

    pages = []

    for page_number, page in enumerate(document, start=1):

        text_blocks = extract_text_blocks(page)

        page_width = page.rect.width

        text_blocks = remove_figure_blocks(
            text_blocks,
            page_width=page_width,
        )


        text_blocks = [
    block
    for block in text_blocks
    if not is_figure_caption(block)
]

        text_blocks.sort(
            key=lambda block: (
                block["y0"],
                block["x0"],
            )
        )

        text = "\n".join(
            block["text"]
            for block in text_blocks
        )

        text = clean_text(text)
        text = clean_page_artifacts(text)

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
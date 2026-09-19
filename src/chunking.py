import re

from nltk.tokenize import PunktSentenceTokenizer


def is_section_heading(line: str) -> bool:
    """Return True when a line looks like an academic section heading."""

    line = line.strip()

    if not line:
        return False

    if line.upper() in {
        "ABSTRACT",
        "INTRODUCTION",
        "CONCLUSION",
        "REFERENCES",
        "ACKNOWLEDGMENTS",
    }:
        return True

    return bool(
        re.match(
            r"^\d+(?:\.\d+)*\s+[A-Z][A-Z0-9\s\-:&()]+$",
            line,
        )
    )


def normalize_section_headings(lines: list[str]) -> list[str]:
    """
    Combine section numbers that PDF extraction places on separate lines.

    Example:
        1
        INTRODUCTION

    becomes:
        1 INTRODUCTION
    """

    normalized = []
    index = 0

    while index < len(lines):
        line = lines[index].strip()

        if (
            re.fullmatch(r"\d+(?:\.\d+)*", line)
            and index + 1 < len(lines)
        ):
            next_line = lines[index + 1].strip()

            if next_line and next_line.isupper():
                normalized.append(f"{line} {next_line}")
                index += 2
                continue

        normalized.append(line)
        index += 1

    return normalized


from nltk.tokenize import PunktSentenceTokenizer




def merge_pdf_lines(
    fragments: list[tuple[str, int]],
) -> list[tuple[str, int]]:
    """
    Merge PDF line fragments that are likely part of the same sentence.
    """

    merged = []

    for text, page in fragments:

        text = text.strip()

        if not text:
            continue

        if not merged:
            merged.append((text, page))
            continue

        previous_text, previous_page = merged[-1]

        # A line ending in sentence punctuation is likely complete.
        if re.search(r"[.!?]$", previous_text):
            merged.append((text, page))
            continue

        # Otherwise, the PDF probably wrapped the sentence.
        merged[-1] = (
            f"{previous_text} {text}",
            previous_page,
        )

    return merged


def split_into_sections(
    pages: list[dict],
) -> list[dict]:
    """
    Split document text into sections and sentences while
    preserving the page number of every sentence.
    """

    sections = []

    current_section = None
    current_fragments = []

    def flush_section():
        if current_section is None or not current_fragments:
            return


        fragments = merge_pdf_lines(current_fragments)

        section_text_parts = []
        page_ranges = []

        current_position = 0

        for fragment_text, page_number in fragments:

            if section_text_parts:
                section_text_parts.append(" ")
                current_position += 1

            start = current_position

            section_text_parts.append(fragment_text)

            current_position += len(fragment_text)

            end = current_position

            page_ranges.append(
                {
                    "start": start,
                    "end": end,
                    "page": page_number,
                }
            )

        section_text = "".join(section_text_parts)

        tokenizer = PunktSentenceTokenizer()

        sentences = []

        for start, end in tokenizer.span_tokenize(section_text):

            sentence_text = section_text[start:end].strip()

            if not sentence_text:
                continue

            sentence_pages = [
                item["page"]
                for item in page_ranges
                if item["start"] < end and item["end"] > start
            ]

            if not sentence_pages:
                continue

            sentences.append(
                {
                    "text": sentence_text,
                    "page_start": min(sentence_pages),
                    "page_end": max(sentence_pages),
                }
            )

        if sentences:
            sections.append(
                {
                    "section": current_section,
                    "sentences": sentences,
                }
            )

    for page in pages:

        lines = [
            line.strip()
            for line in page["text"].split("\n")
            if line.strip()
        ]

        lines = normalize_section_headings(lines)

        for line in lines:

            if is_section_heading(line):

                flush_section()

                current_section = line
                current_fragments = []

                continue

            if current_section is None:
                continue

            current_fragments.append(
                (
                    line,
                    page["metadata"]["page"],
                )
            )

    flush_section()

    return sections

def build_chunks(
    sections: list[dict],
    chunk_size: int = 1000,
    chunk_overlap: int = 1,
) -> list[dict]:
    """
    Build sentence-aware chunks without crossing section boundaries.
    """

    chunks = []

    for section in sections:

        section_name = section["section"]
        sentences = section["sentences"]

        current_sentences = []
        current_length = 0

        for sentence in sentences:

            sentence_length = len(sentence["text"])

            if (
                current_sentences
                and current_length + sentence_length + 1 > chunk_size
            ):
                chunks.append(
                    {
                        "text": " ".join(
                            item["text"]
                            for item in current_sentences
                        ),
                        "section": section_name,
                        "page_start": min(
                            item["page_start"]
                            for item in current_sentences
                        ),
                        "page_end": max(
                            item["page_end"]
                            for item in current_sentences
                        ),
                    }
                )

                if chunk_overlap > 0:
                    overlap_sentences = current_sentences[
                        -chunk_overlap:
                    ]
                else:
                    overlap_sentences = []

                current_sentences = overlap_sentences.copy()

                current_length = sum(
                    len(item["text"])
                    for item in current_sentences
                )

                if current_sentences:
                    current_length += len(current_sentences) - 1

            current_sentences.append(sentence)

            current_length += sentence_length

            if len(current_sentences) > 1:
                current_length += 1

        if current_sentences:
            chunks.append(
                {
                    "text": " ".join(
                        item["text"]
                        for item in current_sentences
                    ),
                    "section": section_name,
                    "page_start": min(
                        item["page_start"]
                        for item in current_sentences
                    ),
                    "page_end": max(
                        item["page_end"]
                        for item in current_sentences
                    ),
                }
            )

    return chunks

def chunk_documents(
    pages: list[dict],
    chunk_size: int = 1000,
    chunk_overlap: int = 1,
) -> list[dict]:
    """
    Convert extracted pages into retrieval-ready chunks.

    Each chunk contains:
        - document_id
        - source
        - page_start
        - page_end
        - section
        - chunk_index
        - chunk_id
    """

    if not pages:
        return []

    source = pages[0]["metadata"]["source"]

    document_id = source.rsplit(".", 1)[0]

    sections = split_into_sections(pages)

    raw_chunks = build_chunks(
        sections,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    chunks = []

    for chunk_index, chunk in enumerate(raw_chunks):

        metadata = {
            "document_id": document_id,
            "source": source,
            "page_start": chunk["page_start"],
            "page_end": chunk["page_end"],
            "section": chunk["section"],
            "chunk_index": chunk_index,
            "chunk_id": f"{document_id}_c{chunk_index:04d}",
        }

        chunks.append(
            {
                "text": chunk["text"].strip(),
                "metadata": metadata,
            }
        )

    return chunks
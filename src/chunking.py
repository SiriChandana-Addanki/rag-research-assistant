from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_documents(
    pages: list[dict],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[dict]:
    """
    Split page-level documents into retrieval-friendly chunks.

    Args:
        pages: Page-level documents produced by PDF ingestion.
        chunk_size: Maximum target size of each chunk.
        chunk_overlap: Number of characters shared between chunks.

    Returns:
        A list of chunks containing text and citation metadata.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n\n",
            "\n",
            ". ",
            "? ",
            "! ",
            "; ",
            ", ",
            " ",
            "",
        ],
        length_function=len,
    )

    chunks = []

    for page in pages:
        page_chunks = splitter.split_text(page["text"])

        for chunk_index, chunk_text in enumerate(page_chunks):
            chunks.append(
                {
                    "text": chunk_text.strip(),
                    "metadata": {
                        **page["metadata"],
                        "chunk_id": (
                            f"{page['metadata']['source']}"
                            f"_p{page['metadata']['page']}"
                            f"_c{chunk_index}"
                        ),
                    },
                }
            )

    return chunks
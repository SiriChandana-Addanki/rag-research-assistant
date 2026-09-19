from src.chunking import chunk_documents


def test_document_chunking():

    pages = [
        {
            "text": (
                "ABSTRACT\n"
                "Retrieval augmented generation combines information "
                "retrieval with language generation. "
                "This approach improves factual accuracy.\n"
            ),
            "metadata": {
                "source": "test.pdf",
                "page": 1,
            },
        },
        {
            "text": (
                "This approach can retrieve useful information "
                "from external sources. "
                "The retrieved information can then ground "
                "the generated answer.\n"
            ),
            "metadata": {
                "source": "test.pdf",
                "page": 2,
            },
        },
    ]

    chunks = chunk_documents(
        pages,
        chunk_size=150,
        chunk_overlap=1,
    )

    assert len(chunks) > 1

    for chunk in chunks:

        assert chunk["text"]

        metadata = chunk["metadata"]

        assert metadata["document_id"] == "test"
        assert metadata["source"] == "test.pdf"

        assert "page_start" in metadata
        assert "page_end" in metadata

        assert metadata["page_start"] <= metadata["page_end"]

        assert metadata["section"] == "ABSTRACT"

        assert "chunk_index" in metadata
        assert "chunk_id" in metadata


def test_chunk_can_span_pages():

    pages = [
        {
            "text": (
                "ABSTRACT\n"
                "Retrieval augmented generation combines "
                "information retrieval with language generation."
            ),
            "metadata": {
                "source": "test.pdf",
                "page": 1,
            },
        },
        {
            "text": (
                "This approach improves factual accuracy "
                "by providing relevant external information."
            ),
            "metadata": {
                "source": "test.pdf",
                "page": 2,
            },
        },
    ]

    chunks = chunk_documents(
        pages,
        chunk_size=500,
        chunk_overlap=0,
    )

    spanning_chunks = [
        chunk
        for chunk in chunks
        if chunk["metadata"]["page_start"]
        != chunk["metadata"]["page_end"]
    ]

    assert spanning_chunks




def test_sentences_are_not_split_by_pdf_line_breaks():

    pages = [
        {
            "text": (
                "ABSTRACT\n"
                "This is a sentence that is broken across "
                "multiple PDF lines and should remain one sentence."
            ),
            "metadata": {
                "source": "test.pdf",
                "page": 1,
            },
        }
    ]

    chunks = chunk_documents(
        pages,
        chunk_size=500,
        chunk_overlap=0,
    )

    assert len(chunks) == 1
    assert (
        "multiple PDF lines and should remain one sentence."
        in chunks[0]["text"]
    )


def test_chunk_page_metadata_is_preserved():

    pages = [
        {
            "text": (
                "ABSTRACT\n"
                "This sentence is on page one."
            ),
            "metadata": {
                "source": "test.pdf",
                "page": 1,
            },
        },
        {
            "text": (
                "This sentence is on page two."
            ),
            "metadata": {
                "source": "test.pdf",
                "page": 2,
            },
        },
    ]

    chunks = chunk_documents(
        pages,
        chunk_size=500,
        chunk_overlap=0,
    )

    assert len(chunks) == 1
    assert chunks[0]["metadata"]["page_start"] == 1
    assert chunks[0]["metadata"]["page_end"] == 2




def test_chunks_do_not_cross_sections():

    pages = [
        {
            "text": (
                "ABSTRACT\n"
                "This is abstract content.\n"
                "INTRODUCTION\n"
                "This is introduction content."
            ),
            "metadata": {
                "source": "test.pdf",
                "page": 1,
            },
        }
    ]

    chunks = chunk_documents(
        pages,
        chunk_size=500,
        chunk_overlap=0,
    )

    sections = {chunk["metadata"]["section"] for chunk in chunks}

    assert sections == {
        "ABSTRACT",
        "INTRODUCTION",
    }

    for chunk in chunks:
        text = chunk["text"]

        if chunk["metadata"]["section"] == "ABSTRACT":
            assert "introduction content" not in text.lower()

        if chunk["metadata"]["section"] == "INTRODUCTION":
            assert "abstract content" not in text.lower()
from src.chunking import chunk_documents


def test_document_chunking():

    pages = [
        {
            "text": (
                "Retrieval augmented generation combines information "
                "retrieval with language generation. "
            )
            * 100,
            "metadata": {
                "source": "test.pdf",
                "page": 1,
            },
        }
    ]

    chunks = chunk_documents(
        pages,
        chunk_size=500,
        chunk_overlap=100,
    )

    assert len(chunks) > 1

    for chunk in chunks:
        assert chunk["text"]
        assert "metadata" in chunk
        assert chunk["metadata"]["source"] == "test.pdf"
        assert chunk["metadata"]["page"] == 1
        assert "chunk_id" in chunk["metadata"]
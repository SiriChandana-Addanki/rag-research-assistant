import json
from pathlib import Path

from src.ingestion import extract_text_from_pdf
from src.chunking import chunk_documents


PDF_PATH = Path("data/raw/paper1.pdf")
DATASET_PATH = Path("evaluation/retrieval_dataset.json")


def load_retrieval_dataset() -> list[dict]:
    with DATASET_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_chunks() -> list[dict]:
    pages = extract_text_from_pdf(PDF_PATH)

    return chunk_documents(pages)


def find_candidate_chunks(
    chunks: list[dict],
    relevant_pages: list[int],
) -> list[dict]:

    candidates = []

    for chunk in chunks:

        chunk_pages = set(
            range(
                chunk["metadata"]["page_start"],
                chunk["metadata"]["page_end"] + 1,
            )
        )

        if chunk_pages.intersection(relevant_pages):
            candidates.append(chunk)

    return candidates


def print_candidates(
    question: dict,
    candidates: list[dict],
) -> None:

    print("\n" + "=" * 100)

    print(f'{question["id"]}: {question["question"]}')

    print(f'Expected pages: {question["relevant_pages"]}')

    print("=" * 100)

    for chunk in candidates:

        metadata = chunk["metadata"]

        print(
            f'\n[{metadata["chunk_id"]}] '
            f'Section: {metadata["section"]} '
            f'Pages: {metadata["page_start"]}-{metadata["page_end"]}'
        )

        print("-" * 100)

        print(chunk["text"])


def main() -> None:

    dataset = load_retrieval_dataset()
    chunks = load_chunks()

    print(f"Total chunks: {len(chunks)}")
    print(f"Total questions: {len(dataset)}")

    for question in dataset:

        relevant_pages = set(
            question["relevant_pages"]
        )

        candidates = find_candidate_chunks(
            chunks,
            relevant_pages,
        )

        print_candidates(
            question,
            candidates,
        )


if __name__ == "__main__":
    main()
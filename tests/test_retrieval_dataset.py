import json
from pathlib import Path


DATASET_PATH = (
    Path(__file__).resolve().parents[1]
    / "evaluation"
    / "retrieval_dataset.json"
)


def test_retrieval_dataset_has_valid_question_level_relevance_labels():
    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))

    assert dataset
    assert len({item["id"] for item in dataset}) == len(dataset)

    for item in dataset:
        assert set(item) == {
            "id",
            "question",
            "relevant_sections",
            "relevant_pages",
        }
        assert isinstance(item["id"], str) and item["id"]
        assert isinstance(item["question"], str) and item["question"]
        assert all(
            isinstance(section, str) and section
            for section in item["relevant_sections"]
        )
        assert all(
            isinstance(page, int) and page > 0
            for page in item["relevant_pages"]
        )

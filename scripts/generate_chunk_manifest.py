"""Generate the tracked chunk-manifest artifact from the local paper corpus."""

import json
from pathlib import Path
import sys


# Direct execution sets sys.path to scripts/, not the repository root.
# Include the root so the existing src package is importable without requiring
# callers to set PYTHONPATH or install the project.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


from src.chunking import chunk_documents
from src.ingestion import extract_text_from_pdf


PDF_PATH = Path("data/raw/paper1.pdf")
MANIFEST_PATH = Path("evaluation/chunk_manifest.json")
MANIFEST_FIELDS = (
    "document_id",
    "chunk_id",
    "chunk_index",
    "page_start",
    "page_end",
    "section",
    "chunk_text",
)


def build_manifest(chunks: list[dict]) -> list[dict]:
    """Project existing chunk metadata and text into a stable JSON manifest."""

    manifest = []

    for chunk in chunks:
        metadata = chunk["metadata"]
        manifest.append(
            {
                "document_id": metadata["document_id"],
                "chunk_id": metadata["chunk_id"],
                "chunk_index": metadata["chunk_index"],
                "page_start": metadata["page_start"],
                "page_end": metadata["page_end"],
                "section": metadata["section"],
                "chunk_text": chunk["text"],
            }
        )

    return manifest


def generate_manifest(
    pdf_path: Path = PDF_PATH,
    manifest_path: Path = MANIFEST_PATH,
) -> list[dict]:
    """Run the existing extraction and fixed-default chunking pipeline."""

    if not pdf_path.is_file():
        raise FileNotFoundError(
            f"Source PDF not found: {pdf_path}. "
            "Provide the ignored local corpus before generating the manifest."
        )

    pages = extract_text_from_pdf(pdf_path)
    chunks = chunk_documents(pages)
    manifest = build_manifest(chunks)

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    return manifest


def main() -> None:
    manifest = generate_manifest()
    print(f"Wrote {len(manifest)} chunks to {MANIFEST_PATH}")


if __name__ == "__main__":
    main()

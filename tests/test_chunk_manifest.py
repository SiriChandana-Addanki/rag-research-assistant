import json
from pathlib import Path
import subprocess
import sys

from scripts.generate_chunk_manifest import (
    MANIFEST_FIELDS,
    build_manifest,
    generate_manifest,
)


def test_generated_manifest_has_required_schema_and_unique_chunk_ids(tmp_path):
    pdf_path = tmp_path / "sample.pdf"
    manifest_path = tmp_path / "chunk_manifest.json"

    import pymupdf

    document = pymupdf.open()
    page = document.new_page()
    page.insert_text(
        (72, 72),
        "ABSTRACT\nA fixture sentence for manifest generation.",
    )
    document.save(pdf_path)
    document.close()

    manifest = generate_manifest(pdf_path, manifest_path)
    written_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert written_manifest == manifest
    assert manifest
    assert len({item["chunk_id"] for item in manifest}) == len(manifest)

    for item in manifest:
        assert tuple(item) == MANIFEST_FIELDS
        assert isinstance(item["document_id"], str) and item["document_id"]
        assert isinstance(item["chunk_id"], str) and item["chunk_id"]
        assert isinstance(item["chunk_index"], int) and item["chunk_index"] >= 0
        assert isinstance(item["page_start"], int) and item["page_start"] > 0
        assert isinstance(item["page_end"], int)
        assert item["page_start"] <= item["page_end"]
        assert isinstance(item["section"], str) and item["section"]
        assert isinstance(item["chunk_text"], str) and item["chunk_text"]


def test_build_manifest_uses_existing_chunk_fields_without_rechunking():
    chunks = [
        {
            "text": "Verified fixture chunk text.",
            "metadata": {
                "document_id": "paper1",
                "chunk_id": "paper1_c0000",
                "chunk_index": 0,
                "page_start": 1,
                "page_end": 1,
                "section": "ABSTRACT",
            },
        }
    ]

    assert build_manifest(chunks) == [
        {
            "document_id": "paper1",
            "chunk_id": "paper1_c0000",
            "chunk_index": 0,
            "page_start": 1,
            "page_end": 1,
            "section": "ABSTRACT",
            "chunk_text": "Verified fixture chunk text.",
        }
    ]


def test_generator_script_runs_from_repository_root_without_import_error():
    repository_root = Path(__file__).resolve().parents[1]

    result = subprocess.run(
        [sys.executable, "scripts/generate_chunk_manifest.py"],
        cwd=repository_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert "ModuleNotFoundError: No module named 'src'" not in result.stderr
    assert "Source PDF not found: data/raw/paper1.pdf" in result.stderr

import pytest

try:
    from ingest import chunk_text
except ModuleNotFoundError:
    pytest.skip("chunk_text tests require ingest dependencies", allow_module_level=True)


def test_chunk_text_overlap():
    text = "Sentence one. Sentence two is longer and continues. Sentence three." * 3
    chunks = chunk_text(text, chunk_size=60, overlap=15)
    assert len(chunks) > 1
    #ensure overlap ensures continuity
    for prev, curr in zip(chunks, chunks[1:]):
        assert len(prev) <= 60
        assert len(curr) <= 60
        #ensure overlap portion appears in next chunk
        overlap_slice = prev[-15:]
        assert overlap_slice.strip() == "" or overlap_slice in curr


def test_chunk_text_no_content():
    assert chunk_text("   \n\t  ") == []

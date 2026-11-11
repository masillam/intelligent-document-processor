import pytest

from idp.loaders import load_file


def test_markdown_loading(tmp_path):
    f = tmp_path / "a.md"
    f.write_text("# Title\nHello world")
    docs = load_file(f)
    assert docs[0].metadata["file_type"] == "md"
    assert "Hello" in docs[0].page_content


def test_unsupported_extension(tmp_path):
    f = tmp_path / "a.xyz"
    f.write_text("x")
    with pytest.raises(ValueError):
        load_file(f)

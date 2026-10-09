"""Internal temporary PDF fixtures validate the maintained renderer; no user document exports."""

import os
import subprocess
import sys
from pathlib import Path

import pytest
from pypdf import PdfReader

from processors.format_converter import FormatConverterProcessor


def test_normal_chromium_preserves_text_table_page_options_and_disables_scripts(tmp_path):
    processor = FormatConverterProcessor({"output_dir": str(tmp_path)})
    output = tmp_path / "controlled.pdf"
    processor._convert_markdown_to_pdf(
        "# Controlled PDF source\n\n| Item | Value |\n| --- | --- |\n| Test | 42 |\n"
        '<script>document.body.innerText = "SCRIPT_EXECUTED";</script>',
        str(output),
    )
    reader = PdfReader(output)
    text = "\n".join(page.extract_text() for page in reader.pages)
    assert "Controlled PDF source" in text and "42" in text
    assert "SCRIPT_EXECUTED" not in text
    assert abs(float(reader.pages[0].mediabox.width) - 595.28) < 1
    assert abs(float(reader.pages[0].mediabox.height) - 841.89) < 1
    processor.pdf_options = {"page-size": "Letter", "orientation": "Landscape", "margin-left": "10mm"}
    processor._convert_markdown_to_pdf("Controlled landscape", str(output))
    assert abs(float(PdfReader(output).pages[0].mediabox.width) - 792) < 1


@pytest.mark.parametrize("reference", ["file:///controlled-secret", "https://example.invalid/controlled"])
def test_local_and_remote_assets_fail_before_output(tmp_path, reference):
    processor = FormatConverterProcessor({"output_dir": str(tmp_path)})
    output = tmp_path / "rejected.pdf"
    with pytest.raises(ValueError, match="local and external resources are disabled"):
        processor._convert_markdown_to_pdf(f'<img src="{reference}">', str(output))
    assert not output.exists()


def test_unknown_dangerous_legacy_options_are_not_silently_accepted(tmp_path):
    processor = FormatConverterProcessor({"output_dir": str(tmp_path), "pdf_options": {"enable-local-file-access": ""}})
    with pytest.raises(ValueError, match="Unsupported PDF options"):
        processor._convert_markdown_to_pdf("Controlled source", str(tmp_path / "rejected.pdf"))


def test_missing_browser_is_a_real_prerequisite_failure(tmp_path):
    environment = dict(os.environ, PLAYWRIGHT_BROWSERS_PATH=str(tmp_path / "empty-browser-directory"))
    output = tmp_path / "missing-browser.pdf"
    program = (
        "import sys; from processors.format_converter import FormatConverterProcessor; "
        "FormatConverterProcessor({'output_dir':sys.argv[1]})._convert_markdown_to_pdf('controlled',sys.argv[2])"
    )
    result = subprocess.run(
        [sys.executable, "-c", program, str(tmp_path), str(output)],
        cwd=Path(__file__).parents[2],
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode != 0
    assert "Executable doesn't exist" in result.stderr
    assert not output.exists()

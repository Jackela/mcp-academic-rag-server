# Maintained entry and verification

`servers.mcp_server_sdk:cli_main` owns the stdio entry. TOML discovers the actual
servers, stores, retrievers and other source packages; console aliases share that
entry. Core public exports load lazily so tool discovery and `--help` do not require
initializing optional RAG backends. MCP SDK v1 is bounded below v2, whose handler API
changed. Python 3.10 is the minimum supported interpreter for current dependencies.

```bash
python -m build --wheel
python -m venv .venv-wheel
.venv-wheel/bin/python -m pip install dist/*.whl
.venv-wheel/bin/python scripts/check_installed_wheel.py
.venv-wheel/bin/python -m unittest discover -s tests/contracts -v
.venv-wheel/bin/python -m unittest discover -s tests/unit -p test_ocr_processor.py -v
```

The install check runs all five console aliases with `--help`, then starts the
installed executable outside the source tree and initializes a real MCP stdio
session. It lists tools, checks the echo and the missing-document error. It supplies
a syntactically valid fixture key; no live credential, OCR or model request is made.
The contract suites cover actual `OCRProcessor.process(Document)`, not the stale
`initialize`/`process_pdf_file` test interface that the implementation never exposed.
They check page Markdown, connector payloads, upload/signed URL failures, empty
text and partial-file failures. The processor does not report empty/partial OCR as success.

The Mistral connector uses the current `/v1/ocr`, `/v1/files`,
`/v1/files/{file_id}/url` contracts and `pages[].markdown`, with old `text` page fields
retained for compatibility. Upload streams close after requests and multipart
Content-Type is left to requests. Sources reviewed 2026-10-09:
[Mistral OCR API](https://docs.mistral.ai/api/endpoint/ocr),
[Mistral Files API](https://docs.mistral.ai/api/endpoint/files), and
[MCP SDK migration](https://github.com/modelcontextprotocol/python-sdk/blob/main/docs/migration.md).

The package contract workflow now runs real install/protocol/OCR tests. The broader
CI, storage and documentation workflows retain their checks. Their type/security
failures are no longer discarded. Actionlint checks all workflow syntax; service,
format, type and behavioral results are recorded per actual run. This repair does
not certify remote models, vector services, full RAG quality or a production release.

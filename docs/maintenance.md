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

## Full-suite recovery evidence

On 2026-10-09, Python 3.11 with the declared dependencies passed the complete
unit suite (364 tests, 4 subtests). The five existing intentionally failing CI
probe cases remain disabled by their own opt-in fixture. FAISS-unavailable is
now tested explicitly even when FAISS is installed. Native vector updates use
new Haystack documents instead of mutating deprecated fields. Context cleanup
uses the manager's public sessions. MIME validation fails closed when libmagic
cannot determine a type, and path checks resolve the base and reject symlinks.
Tests cover real configuration removal/persistence, processor discovery,
Haystack run/messages and public session queries; nonexistent historical APIs
are not implemented to satisfy stale fixtures.

Every cleanup entry consumes only explicitly registered resources. PID cleanup
checks both parent ownership and creation time. The contract creates two same-name
child processes itself: the registered child exits while the unregistered control
stays alive. It also proves an unregistered file is preserved and no broad name
or port scanning command runs. User and other agent processes are never test subjects.

HTML documentation builds with warnings as errors; RST checking delegates only
Sphinx extension directives to that real Sphinx build. Markdown remains editable
source. Missing guide chapters and deployment materials are marked as missing.
The old global lint/type debt remains a separate failing check until repaired;
these behavioral results do not establish a complete CI pass.

Pytest's default paths match the automated suites actually run in CI. The existing
`tests/manual` scripts remain separate live-service experiments; they are not
silently treated as verified research or added to offline CI by discovery tests.
Explicit manual execution and its service/data requirements remain unverified.

A second recovery pass exercises the current Haystack 2.x embedders/retrievers,
real in-memory and FAISS stores, prompts, sessions and migration without service
calls: 77 targeted integration/component/unit cases and 9 MCP boundary cases pass.
Three negative migration cases have equal document counts but changed content,
metadata or vectors; verification rejects each. Memory storage still has no disk
backup implementation, so a requested backup fails rather than reporting success.
Its migration test explicitly opts out of backup; source documents remain intact
when backup fails. All 11 existing performance cases pass with correctly sized
fixture batches and vector dimensions. These are local controlled benchmarks,
not host deployment performance or research-quality evidence.

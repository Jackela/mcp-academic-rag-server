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


## Current parser, storage and coverage boundaries

`Document.get_text_content()` owns conversion of current OCR text dictionaries,
embedding text chunks and historical string stages into stored text. Embedding
and storage consume it together. Offline tests cover current and legacy inputs,
partial batches and a failing embedding boundary without model or OCR requests.

Math markup uses the current Python-Markdown InlineProcessor/BlockProcessor APIs
and `xml.etree.ElementTree`; tests parse the actual rendered HTML and reject
nontext OCR values. Internal temporary PDF tests use normal Chromium with scripts disabled,
verify actual text, tables and page dimensions, reject local/remote resources and dangerous
legacy options, and prove a missing browser fails. They do not export a user deliverable or
establish extraction quality. PDF creation requires `python -m playwright install chromium`.
[Python-Markdown extension API](https://python-markdown.github.io/extensions/api/)
explains these parser contracts.

Vector factory migration delegates to `VectorStoreMigrator.migrate`, performs the
copy and verifies real target documents. Its historical no-op success path is
removed. The compatibility helper has no backup parameter and explicitly performs
migration without backup; the migrator's separate requested-backup failure stays
observable. Milvus availability requires an actual backend class, not just an
installed client dependency. The repository's Milvus document-store stub reports
unavailable; installing its SDK does not implement that stub.

Vector CI measures the complete `document_stores` directory, including executed
migration lines. It does not additionally name the dotted migration submodule:
coverage's early lookup imported native Torch a second time and caused a locally
reproduced exit 139. No coverage files or valid test cases are excluded by removing
that duplicate. See [Coverage source selection](https://coverage.readthedocs.io/en/latest/source.html).

## Monitoring runtime and provider boundary

`pyproject.toml` owns runtime and development dependency constraints; `requirements.txt`
installs the declared development and monitoring extras. MCP-only wheel installation
keeps monitoring optional. The monitoring HTML source is `core/templates/dashboard.html`;
dashboard initialization only reads this packaged template. `scripts/check_monitoring_wheel.py`
checks actual HTML and metrics/alerts/health HTTP routes with a read-only installed package.

Telemetry requires the actual OpenTelemetry SDK when initialized. Disabled tracing or metrics
use the SDK API's explicit no-op providers; a missing SDK or explicitly enabled missing
exporter fails initialization and leaves no initialized singleton. Legacy Jaeger Thrift imports
are loaded only when explicitly selected. Modern Jaeger supports OTLP, as documented by
[OpenTelemetry](https://opentelemetry.io/docs/compatibility/migration/).
`PrometheusMetricReader` registers a collector but does not host HTTP; the obsolete `endpoint`
constructor option is rejected explicitly. Hosts configure their HTTP server separately, following
the [Python exporter documentation](https://opentelemetry.io/docs/languages/python/exporters/).
SDK tests collect actual in-memory spans and metric data without external telemetry services.

RAG invokes the existing `BaseLLMConnector.generate` contract through a typed Haystack component,
so providers with client-only implementations do not need a nonexistent native `generator`.
Per-call generation options reach that execution path; OpenAI native messages use `ChatMessage.text`.
Controlled provider fixtures validate retrieval, options, native reply text and explicit provider errors;
normal browser tests retain the production HTTP and JavaScript flow.

## Security checks

The old Safety command passed a filename to an output-format enum and failed before scanning.
The current Safety CLI also requires account-backed service initialization. CI audits the actually
installed project dependency environment with [PyPA pip-audit](https://github.com/pypa/pip-audit),
which reports known vulnerabilities as JSON and preserves a nonzero failure status. No vulnerability
is ignored. Semgrep runs in a separate tools environment to keep its own OpenTelemetry constraints
from changing the project runtime; `--error` makes findings fail the gate. Bandit scans the declared
source directories with the repository's existing `pyproject.toml` configuration, avoiding temporary
virtualenv copies while retaining all configured source checks. Publishing validation runs argument
lists without a shell and aborts on a failed artifact check; validation did not publish a package.

PDF parsing uses maintained `pypdf`; the deprecated PyPDF2 vulnerability advisory
[requires migration](https://github.com/pypa/advisory-database/blob/main/vulns/pypdf2/PYSEC-2026-1835.yaml).
The unpatched [pdfkit advisory](https://github.com/pypa/advisory-database/blob/main/vulns/pdfkit/PYSEC-2026-2860.yaml)
is addressed by removing that renderer and its dependency. Chromium preserves page size,
orientation, UTF-8 and margins. Unsupported wkhtmltopdf-specific options fail explicitly.
NLTK had no source consumer and was removed from the speculative enhanced declaration;
its unpatched model-artifact advisory is not ignored. A fresh installation of declared
`.[dev,monitoring]` was audited: 185 distributions, no known vulnerabilities.

`scripts/check_bandit.py` validates and applies the existing medium severity and medium
confidence configuration with `-ll/-ii`, preserving the existing B101/B601 skips. It also saves
all severities/confidences to `bandit-report-all.json`; low findings remain visible. A malformed
or incomplete scan fails. The former command did not apply configured thresholds and scanned
temporary virtualenv copies. Actual clean, low-informational and high-failing fixture sources
verify scanner statuses; no finding is converted to success with a shell fallback.

## Retained and retired server entry contracts

The config-center example uses its validated effective `ConfigManager` snapshot through
`ServerContext(config_manager=...)`, then delegates processing/query execution to the
maintained SDK helpers with explicit context injection. Its old content/id/embedding input
schema referred to methods and constructors that do not exist; current document tools take
`file_path`, preserve model-stage metadata and share the actual native RAG store/session.
The retained debug servers use one JSON line transport, terminate on stdin EOF, reject
malformed JSON/requests/params, ignore notifications and retain the current request ID on failure.
Empty PDF text does not become an extraction success or fictitious OCR fallback.

`tests/temp_root_tests/test_mcp.py` was retired: it hardcoded an inaccessible E: drive path
and awaited a response to initialization notifications, which can hang. The actual official
MCP stdio client contract lives in `tests/e2e/test_rag_workflow.py`; it verifies discovery,
processing and registered logical failures with native `CallToolResult.isError=True`.
Helper list APIs remain compatible, and success after a controlled failure is checked.

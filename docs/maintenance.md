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

On 2026-10-09, before the Windows MIME/platform-type follow-up, the frozen
`bf4ba464` Python 3.11 candidate passed all default unit, contract,
integration, component, E2E and performance suites: 668 tests and 22 subtests.
All 197 Python source hashes were unchanged between the run's start and end.
Native startup used the selected Torch OpenMP library with the bypass disabled.
The five existing intentionally failing CI probe cases remain disabled by their
own opt-in fixture. FAISS-unavailable is
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
The final local formatting, lint and type checks pass across the declared targets;
the corresponding remote workflow results remain separate evidence. These behavioral
results alone do not establish a complete CI pass.

Pytest's default paths match the automated suites actually run in CI. The existing
`tests/manual` scripts remain separate live-service experiments; they are not
silently treated as verified research or added to offline CI by discovery tests.
Explicit manual execution and its service/data requirements remain unverified.

## Container entry and verification

The image installs the package from `pyproject.toml` rather than installing a development
requirements file outside the checkout. The old builder copied only `requirements.txt`,
whose editable project reference could not resolve. Its floating Debian image also named
obsolete system packages, and its runtime referenced missing root `mcp_server.py` and
`health_check.py` files. `EXPOSE 8000` did not create an HTTP server.

The maintained image uses Python 3.11 on Debian Bookworm, selects official CPU Torch wheels,
and runs `mcp-academic-rag-server` as a non-root user. No HTTP health check is declared.
Startup environment validation checks key syntax and creates the configured data directory;
it does not verify OCR, model availability or answer quality. Default browser rendering is
not installed in the image: hosts that need PDF rendering install Chromium and system
dependencies in a derived image. The existing explicit missing-browser error remains active.

```bash
docker build -t academic-rag:local .
python scripts/check_container.py academic-rag:local
```

This check uses an actual MCP `ClientSession` against a unique, owned `docker run -i`
container with networking disabled. It validates initialization, discovery, echo, native
missing-file/invalid-arguments errors, non-root execution, native Torch/FAISS operations,
installed dependency consistency, help and rejected missing credentials. Cleanup only
addresses that exact test container. CI loads the built image before this check; it no longer
waits thirty seconds before invoking a nonexistent health script. The historical Compose
stack remains separate and unverified: it assumes HTTP on port 8000 and references absent
`config/nginx.conf`. It is not the current MCP stdio client configuration.

Primary installation references: [PyTorch CPU installation](https://pytorch.org/get-started/locally/)
and [Playwright Docker prerequisites](https://playwright.dev/python/docs/docker).

## Publication is an explicit action

Ordinary pushes and pull requests run checks without publishing images. Manual workflow
dispatch defaults `publish_target` to `none`; explicit `staging` or `production` selection
publishes the corresponding image only after the required checks. These jobs publish
container images and do not claim a running website or deployment; the unsupported example
domain URLs and run-number version tags are removed. Manual images use the existing commit
SHA, while an explicitly published GitHub release keeps its actual existing tag.

Release checkout fetches full history. `scripts/release_changelog.py` handles first releases
with no earlier tag and rejects missing current tags. Git commit subjects remain text in a
random-delimited environment value and are read with `process.env.CHANGELOG`; they are never
inserted into JavaScript source. Temporary local Git fixtures check literal shell/JavaScript
syntax remains data. The conditions test ordinary pushes/PRs, default manual checks, explicit
manual targets and published/created release events without publishing an artifact.

The retained Docker storage job now starts the same real embedded Milvus service declared
in `.github/milvus-ci.yml`; its old independent service omitted the standalone command,
polled the gRPC port with HTTP, and selected a nonexistent `requires_milvus` test set.
The replacement `scripts/check_milvus_service.py` uses the actual optional `storage-tests`
SDK declaration in an isolated service environment. Local Linux ARM Milvus 2.3.4 and
PyMilvus 2.6.17 passed real two-vector insertion, nearest-neighbor ordering, row query and
owned-collection cleanup. The script rejects non-loopback hosts before connecting; the
local fixture starts and removes only its own uniquely named container on an ephemeral
loopback port. This verifies service protocol compatibility, not the unimplemented
application `MilvusDocumentStore`. SDK deprecation warnings remain visible in its log.
See [Milvus SDK compatibility](https://milvus.io/api-reference/pymilvus/v2.4.x/About.md).

The MCP `query_documents.top_k` parameter now reaches `ChatSession` and the actual Haystack
retriever. Previously it only sliced the displayed sources while all retrieved text still
entered the prompt. Controlled native retrieval checks ranking, returned counts, prompt
contents, configured-default preservation and invalid limits failing before session/query
creation; other provider/generation options are unchanged.

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

## Darwin native startup

The installed macOS ARM wheels can include separate OpenMP libraries in Torch,
FAISS and scikit-learn. Actual Torch and FAISS operations aborted with exit 134
in both source and installed-wheel environments. Import order alone did not fix
this. Importing scikit-learn implicitly set `KMP_DUPLICATE_LIB_OK=True`, which
masked the conflict in earlier pytest runs; those runs do not prove safe native
interoperation. The repair never enables that bypass or modifies wheel binaries.

The five console aliases and `python -m servers.mcp_server_sdk` prepare the
startup environment after argument parsing and before asynchronous startup.
On Darwin they replace only the explicitly invoked CLI once, preserving
`sys.orig_argv`, stdin, stdout, stderr and exit status. The new process places
its own installed Torch `lib` directory first in `DYLD_LIBRARY_PATH` and fixes
`KMP_DUPLICATE_LIB_OK=FALSE`. A checked reentry marker prevents recursive
replacement. `--help` and `--validate-only` retain their existing lightweight behavior.
Other platforms perform no process replacement or environment changes.

Library imports do not replace a hosting application. An embedded consumer must
launch its Python process with the environment returned by
`utils.native_runtime.native_subprocess_environment()`, then call the SDK's
async `main()`. Applying that mapping to `os.environ` inside an already running
Python process does not configure dyld's startup search path. The SDK fails
startup explicitly if its native preconditions cannot be verified. The helper
does not expose credentials or write a global environment configuration.
The macOS-only guard queries its own process's `KERN_PROCARGS2`, extracts only
the two required startup keys after skipping the counted arguments, and checks
the current settings and already loaded OpenMP paths. Apple marks this kernel
interface `API_UNSTABLE`; it is verified on the current host, and unsupported
or malformed responses fail startup explicitly. This checks ordinary startup
configuration, not a hostile application's deliberate modification of its
initial process stack. Raw arguments and environment values are never logged.

`scripts/check_native_runtime.py` runs in a cold process with the bypass
explicitly disabled. It compares Torch matrix products, actual FAISS insertion
and nearest-neighbor searches, and scikit-learn clustering against independent
NumPy values. On Darwin it also checks that only the selected Torch OpenMP
library is loaded. This check is separate from MCP initialization/discovery and
from external model or OCR requests.

## Security checks

Windows CPython's actual matrix crashed during `python-magic 0.4.27` import in
`magic.compat`, before collecting any test node. Both Windows native numerical
checks had passed. The upstream [Git Bash/MSYS report](https://github.com/ahupp/python-magic/issues/288)
describes the same hang/access-violation pattern; the failing run did not record
which DLL it selected, so that path is not asserted as a measured fact.

Windows development/Web extras select [python-magic-standalone 0.4.28](https://pypi.org/project/python-magic-standalone/0.4.28/),
an explicitly unofficial distribution of the [maintainer-contributed wheel branch](https://github.com/ahupp/python-magic/pull/294).
The wrapper retains its upstream MIT license; bundled native components retain
their upstream licenses. Project version and license metadata are unchanged.
Other platforms retain `python-magic` and their existing libmagic installation.
The Windows adapter binds only the distribution-owned native DLL and database
through absolute paths, verifies pointer architecture and library version,
and searches dependencies only in the owned DLL directory and System32. It does
not import the fork's compatibility/fallback loader or search PATH/CWD for
Cygwin/MSYS libraries. Competing magic distributions, missing files, wrong ABI
and identification errors fail closed. Each request owns and closes its native
handle; Python opens Unicode filenames and supplies bounded content bytes to
the native MIME API. The existing size limit and allowed MIME types remain the
authority. No extension-based acceptance or new MIME inference engine is added.

The real Windows matrix checks seven generated content cases (PDF, PNG, JPEG,
TIFF, UTF-8 text, UTF-16 text and DOCX), Unicode filenames, 21 threaded inspections
and rejection of unknown bytes, a disguised executable, HTML, JSON and shell
content. It emits the actual owned DLL path/hash/version only after assertions
pass. Non-Windows runs explicitly skip that platform acceptance case; local
negative fixtures do not establish Windows compatibility. The subsequent
source-type repair preserves the runtime platform checks while allowing strict
mypy to check the Darwin helper from Linux, Darwin and Windows targets.

The subsequent Windows run collected 496 tests successfully and passed the
native numerical check, but stopped at the existing maximum of ten failures
before reaching the MIME content case. Its failures exposed open temporary-file
handles and default-codepage fixture reads, plus the configuration CLI's Chinese
output on redirected CP1252 streams. Fixtures now close owned temporary handles
and use explicit UTF-8; the CLI configures only its own stdout/stderr as UTF-8.
Controlled CP1252 child processes check real generation, validation and failure
statuses. Windows now runs the same native MIME content case before the complete
unit suite; all unit selections, coverage and failure limits remain unchanged.

The old Safety command passed a filename to an output-format enum and failed before scanning.
The current Safety CLI also requires account-backed service initialization. CI audits the actually
installed project dependency environment with [PyPA pip-audit](https://github.com/pypa/pip-audit),
which reports known vulnerabilities as JSON and preserves a nonzero failure status. No vulnerability
is ignored. The factory-import follow-up passed 47 factory/native-storage cases
including three added negative import cases. These reject unknown backend/module/dependency
inputs before executing their controlled fixture modules. The three maintained backend modules
and two optional dependency imports now use literal module names; fallback and unavailable
Milvus behavior remain intact. Actual Semgrep 1.180.0 auto rules scanned the same 10 source files
with 290 applicable rules: the previous two blocking variable-import findings are zero after
the change, with no rule exclusions. Semgrep runs in a separate tools environment to keep its own OpenTelemetry constraints
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
its unpatched model-artifact advisory is not ignored. The final isolated declared
`.[dev,monitoring]` inventory has 186 distributions: 185 dependencies were audited with no
known vulnerabilities. The local unpublished project itself has no PyPI advisory record
and is explicitly marked unauditable in the JSON report; source checks remain separate.
Redis is installed only in the development extra to check the existing optional adapters
against its actual typed SDK. MCP-only runtime installation retains missing-Redis fallback.

The hosted runner's preinstalled setuptools 79.0.1 produced the actual remote audit failure.
An isolated audit of that exact version reproduces the nonzero status; its two report entries
share `PYSEC-2026-3447`, fixed in setuptools 83. CI now creates a fresh project environment
and upgrades its build tools before installing the declared dependencies, rather than
auditing unrelated or stale hosted-toolcache packages. Failed reports are uploaded as well.

The actual container Trivy scan initially failed before scanning because the
GitHub repository owner had uppercase letters. Each image-building job now
normalizes the same repository name before use; five Bash environment-file
contracts failed before the change and pass afterward. The final Linux ARM
image's seven installed configuration/query/factory/startup source hashes match
the frozen source. Its base Python tooling is upgraded separately from the
isolated runtime. The actual inventories audit four base distributions and 106
public runtime distributions with zero known advisories. The unpublished project
and the CPU Torch local-version distribution are explicitly unauditable on PyPI.

Successful Trivy execution does not mean zero findings. The complete final SARIF
retains 267 Debian advisories with no fixed version reported (2 critical, 53 high,
109 medium, 102 low, 1 unknown) and six Python advisories from dependency-provided
third-party SBOM records. Those six records reference historical msgpack,
setuptools and urllib3 versions without installed metadata paths; the actual
runtime/base metadata has the patched versions. Neither the SBOM records nor
the OS findings are deleted or suppressed. Container publication still follows
the existing explicit release/manual conditions.

The earlier Windows matrix stopped during pytest collection after FAISS loaded;
no test node had started. Its incomplete jobs are not passing evidence. The
native verification and unit steps now have five- and twenty-minute failure
limits, respectively. A C-level repeating traceback watchdog starts before
pytest import/collection and writes through a duplicated original stderr handle,
so pytest capture cannot hide a stalled collector's trace. All test selections,
coverage and the existing 300-second per-test timeout remain intact. Original
logs are uploaded even on failure; missing JUnit output is not fabricated.

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

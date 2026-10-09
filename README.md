# MCP Academic RAG Server

An experimental MCP server for academic document OCR and RAG. The maintained
console entry is `servers.mcp_server_sdk:cli_main`. All five console aliases use
that implementation. Source variants under `servers/` remain in the repository;
they are not separate production releases.

Python 3.10 or newer is required. The installation contract uses MCP SDK v1
(`mcp>=1,<2`) and Haystack 2 (`haystack-ai>=2,<3`). Package version, author and
license metadata remain in [pyproject.toml](pyproject.toml).

## Install and start

From this checkout, use an isolated environment:

```bash
python -m venv .venv
.venv/bin/python -m pip install .
.venv/bin/mcp-academic-rag-server --help
```

`--help` works without credentials or loading optional RAG backends. Set
`OPENAI_API_KEY` securely before using the stdio server. The current entry checks
the configured key's shape; this does not verify the key with an external service.

```bash
.venv/bin/mcp-academic-rag-server --validate-only
.venv/bin/mcp-academic-rag-server
```

The second command waits for an MCP client on stdin/stdout. It does not start an
HTTP dashboard. `--validate-only` checks the environment and creates `DATA_PATH`
(default `./data`), without making a model request. Missing or malformed keys
produce a nonzero exit status. This entry accepts `--help` and `--validate-only`;
logging is controlled by `LOG_LEVEL`.

Configure an MCP client with the absolute path of the installed executable and
provide credentials through its environment configuration. Do not put a live key
in the repository. Document processing and queries also require the applicable
processor, model and storage configuration; use [config/config.json.example](config/config.json.example)
as a source reference rather than treating an environment check as full setup.

## Current MCP tools

The tool list is defined in [servers/mcp_server_sdk.py](servers/mcp_server_sdk.py).

| Tool | Input | Behavior |
| --- | --- | --- |
| `test_connection` | Optional `message` | Echo and current environment information |
| `validate_system` | No arguments | Report configuration and component readiness |
| `process_document` | Required `file_path`, optional `file_name` | Run the configured document pipeline; report errors when unavailable |
| `query_documents` | Required `query`, optional `session_id` and `top_k` | Use the configured RAG pipeline and session; report unavailable components or errors |

OCR uses the Mistral file upload, signed URL and OCR contracts in
[connectors/api_connector.py](connectors/api_connector.py). Empty text, partial
file results and service errors do not count as successful extraction. Local
verification uses controlled fixture text and vectors; no paper extraction,
research finding or model answer quality is claimed from those fixtures.

Markdown-to-PDF conversion requires local Chromium (`python -m playwright install chromium`).
It disables scripts and local/remote resource fetching; self-contained data images are supported.
Missing browsers and unsupported legacy PDF options are explicit errors.

## Develop and verify

[AGENTS.md](AGENTS.md) defines maintenance boundaries. [docs/maintenance.md](docs/maintenance.md)
contains the repeatable installation, cleanup, behavior and documentation checks.
`pyproject.toml` owns package metadata and dependencies, `pytest.ini` owns test
selection, and `.flake8` owns lint settings. Do not duplicate these configurations
in another file.

```bash
.venv/bin/python -m pip install -e '.[dev,monitoring]'
.venv/bin/python -m pytest
```

Default discovery includes unit, contract, integration, component, E2E and
performance tests. Browser tests require the normal Chromium installation:

```bash
.venv/bin/python -m playwright install chromium
.venv/bin/python -m pytest tests/e2e
```

Manual scripts that can request live models or services are outside default
collection. Review their inputs and service requirements before running them.
The existing CI workflows retain formatting, typing, security, documentation and
behavior checks. Read the actual run result and failing command before claiming
that the complete project passes.

## Status and scope

Installation/protocol checks, offline behavior, remote CI, user acceptance and
production release are separate evidence. This repair does not establish live
OCR accuracy, answer quality, deployment readiness, performance targets or
whole-project coverage. Existing Markdown guides under [docs/](docs/) provide
implementation context; use the maintained entry and checks above for current
execution.

The project declares MIT license metadata. A repository `LICENSE` file is absent;
this repair preserves the declaration and does not create a new license grant.

## Acknowledgments

- [Model Context Protocol](https://modelcontextprotocol.io) for the protocol
- [Haystack](https://haystack.deepset.ai/) for RAG components
- [FAISS](https://faiss.ai/) for local vector search
- [OpenAI](https://openai.com/) for model integration
- [OpenTelemetry](https://opentelemetry.io/) for observability components

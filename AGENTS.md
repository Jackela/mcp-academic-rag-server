# Maintenance contract

- Read README.md, pyproject.toml and the relevant source before changing behavior. Preserve published version, author and license metadata.
- Maintained runtime: servers/mcp_server_sdk.py, core/server_context.py, processors/ocr_processor.py and connectors/api_connector.py. Keep public interfaces and errors explicit.
- Markdown is the editable documentation source; JSON/TOML define configuration. Do not claim generated text, local tests or historical reports establish acceptance.
- Start with the relevant offline contract tests in docs/maintenance.md; validate the built wheel outside the checkout and record its exact revision.
- Do not replace service/API behavior with fabricated PDF text, embeddings or research results. Local fixture outputs are test evidence only.
- Keep independent modules small and public compatibility aliases on one maintained entry point. Do not suppress failures to make CI pass.
- Use live OCR/model services or deployment only within explicit task authorization. Record external checks separately from offline checks.

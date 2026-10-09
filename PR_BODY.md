## Summary

Restore the installed MCP stdio entry and current OCR/Haystack/provider execution contracts. Preserve package version, author and license metadata; all five aliases use one maintained entry.

```text
installed MCP entry
  validate arguments / configuration
  native CallToolResult (logical failures set isError)
  shared ServerContext → document pipeline / ChatSession
  per-call top_k → actual retriever → prompt
```

The repair fixes actual shared-store/session behavior, immutable configuration snapshots and atomic saves, owned-resource cleanup, normal browser interaction, packaged monitoring templates, maintained PDF rendering, and explicit unavailable-feature errors. TOML/configuration files own dependencies and runtime settings; Markdown documents the real entry and verification boundaries.

## Evidence

- **Before:** installed aliases and Docker referenced broken or absent entry paths, a nonexistent HTTP health endpoint, and helper-list assertions against native MCP results. **After:** all aliases and the non-root installed CPU image pass actual official MCP initialize/discovery/echo/error contracts with networking disabled, native Torch/FAISS checks and `pip check`. Installed configuration/query source hashes match the frozen candidate.
- **Before:** `top_k` only sliced displayed sources; excluded retrieval text still entered the model prompt. Six real cases failed. **After:** 58 related cases and 4 subtests pass, including native ranked retrieval, actual prompt exclusion, configured defaults and invalid limits failing before query/session creation.
- **Before:** failed configuration saves could corrupt existing files or report successful repair. **After:** same-directory atomic replacement preserves modes and prior files on serialization/replacement failure; the actual repaired status is revalidated (27 focused cases pass).
- **Before:** old broader CI failed at native-result assertions, Vector mypy package resolution, missing declared Redis SDK types, hosted setuptools advisories and a broken Milvus container command/health probe. **After:** 74 source files pass strict mypy, 194 Python files pass formatting and lint, all workflow syntax passes actionlint; declared isolated dependencies audit 185 public distributions with zero known vulnerabilities. The unpublished local project is explicitly unauditable; no vulnerability exclusions were added. Real loopback Milvus 2.3.4 / PyMilvus 2.6.17 insertion, ranking, row query and owned cleanup pass.
- Final complete Python 3.11 discovery: **640 passed, 5 existing failure-probe skips, 22 subtests passed**. HTML documentation builds with warnings as errors and RST checking passes.
- Existing Bandit medium/medium thresholds remain enforced; all 13 LOW findings remain visible in the full report. Normal Chromium Web/PDF, native in-memory telemetry and installed read-only dashboard behavior are verified separately from external services.

The latest complete remote workflow results remain pending at this local evidence snapshot. Earlier failed/interrupted runs are retained as failures; local success does not substitute for the final Python/OS matrix, Docker/security and documentation jobs.

## Merge Danger

**Door:** two-way. **Blast radius:** installed entry, configuration, provider/store/session, monitoring, maintenance and verification paths. Optional features and logical failures now fail explicitly. Historical Compose HTTP/Nginx configuration remains unverified, and the application Milvus store remains unavailable.

Ordinary push/merge does not publish a product container. Manual dispatch defaults to checks only; staging/production requires explicit selection, and a published release retains its existing tag. Documentation Pages retains its existing main-update behavior. This PR does not perform live OCR/model calls, verify research quality, publish product images or establish an actual application deployment.

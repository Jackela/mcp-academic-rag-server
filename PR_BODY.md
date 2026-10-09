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
- **Before:** old broader CI failed at native-result assertions, Vector mypy package resolution, missing declared Redis SDK types, hosted setuptools advisories and a broken Milvus container command/health probe. **After:** 76 source files pass strict mypy, 199 Python files pass formatting and lint, all workflow syntax passes actionlint; declared isolated dependencies audit 185 public distributions with zero known vulnerabilities. The unpublished local project is explicitly unauditable; no vulnerability exclusions were added. Real loopback Milvus 2.3.4 / PyMilvus 2.6.17 insertion, ranking, row query and owned cleanup pass.
- Frozen `bf4ba464` Python 3.11 discovery before the Windows MIME/platform-type follow-up: **668 passed, 5 existing failure-probe skips, 22 subtests passed**; all 197 Python source hashes are unchanged across that run. HTML documentation builds with warnings as errors and RST checking passes. A prior mixed-source run's browser-load timeout is retained; both affected viewport checks and the complete run pass without changing timeout/retry settings.
- **Factory import follow-up:** actual remote Semgrep found two variable imports. Three execution-marker negative cases failed before the repair; 47 factory/native-storage cases now pass. Literal selection preserves the three existing backends and fallback; unknown module/dependency inputs are rejected. The same Semgrep 1.180.0 auto scan runs 290 applicable rules on 10 files with zero findings, without exclusions.
- **Darwin native startup:** actual Torch/FAISS operations aborted with exit 134; scikit-learn's implicit duplicate-OpenMP bypass masked the conflict in earlier pytest runs. The CLI now replaces its process once before startup, preserving argv/stdio and selecting its installed Torch runtime with the bypass explicitly disabled. Embedded SDK imports never replace the host; late environment updates fail before native operations. Eighteen startup/parser contracts, all five installed aliases' official MCP stdio checks, and independent installed SDK/factory/RAG/Torch/FAISS/scikit-learn numerics pass with one OpenMP library and `KMP_DUPLICATE_LIB_OK=FALSE`.
- Existing Bandit medium/medium thresholds remain enforced; all 39 LOW findings in the complete declared CI scope remain visible. Normal Chromium Web/PDF, native in-memory telemetry and installed read-only dashboard behavior are verified separately from external services.
- **Container scan follow-up:** uppercase repository ownership previously stopped Trivy before scanning. All five image jobs normalize the existing name; their real Bash contracts pass. Fresh Trivy now completes and preserves its SARIF: 267 Debian advisories have no fixed version reported, and six historical Python entries originate in dependency-provided SBOM records rather than actual installed metadata. Actual upgraded base tooling and public runtime inventories audit cleanly; the unpublished project and CPU Torch local version are explicitly unauditable. No report entries or scan thresholds are suppressed.
- **Windows follow-up:** both actual native numerical gates pass, but unit collection crashes in the old magic compatibility loader. Windows extras now select the maintained standalone wheel; a small adapter binds only its owned native DLL/database with verified architecture and restricted dependency search. Per-call handles, bounded reads and the existing MIME allowlist fail closed. Local contracts pass; the real Windows seven-content/Unicode/threaded/rejection gate and latest complete candidate checks remain pending. Strict typing passes for all 76 source files under Linux, Darwin and Windows targets without exclusions.

The latest complete remote workflow results remain pending at this local evidence snapshot. Earlier failed/interrupted runs are retained as failures; local success does not substitute for the final Python/OS matrix, Docker/security and documentation jobs.

## Merge Danger

**Door:** two-way. **Blast radius:** installed entry, configuration, provider/store/session, monitoring, maintenance and verification paths. Optional features and logical failures now fail explicitly. Historical Compose HTTP/Nginx configuration remains unverified, and the application Milvus store remains unavailable.

Ordinary push/merge does not publish a product container. Manual dispatch defaults to checks only; staging/production requires explicit selection, and a published release retains its existing tag. Documentation Pages retains its existing main-update behavior. This PR does not perform live OCR/model calls, verify research quality, publish product images or establish an actual application deployment.

---
description: Run MCP Academic RAG tests with hard cleanup
allowed-tools: Bash(python test*), Bash(python -m pytest*), Bash(npx kill-port:*), Bash(ps:*), Bash(pkill:*), Bash(lsof:*), Bash(tasklist:*), Bash(taskkill:*), Bash(netstat:*)
---

## Preflight
Clean environment before testing:
!`$CLAUDE_PROJECT_DIR/.claude/hooks/cleanup.sh`

## Run
Execute comprehensive RAG functionality tests:
!`python test_rag_complete.py`

## Postflight  
Ensure all test processes and ports are cleaned:
!`$CLAUDE_PROJECT_DIR/.claude/hooks/cleanup.sh`
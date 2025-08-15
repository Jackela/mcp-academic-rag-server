---
description: Run quick API validation tests with cleanup
allowed-tools: Bash(python test*), Bash(npx kill-port:*), Bash(ps:*), Bash(pkill:*), Bash(tasklist:*), Bash(taskkill:*), Bash(netstat:*)
---

## Preflight
Clean environment:
!`$CLAUDE_PROJECT_DIR/.claude/hooks/cleanup.sh`

## Run
Execute quick API tests:
!`python test_quick_real_api.py`

## Postflight
Clean up test artifacts:
!`$CLAUDE_PROJECT_DIR/.claude/hooks/cleanup.sh`
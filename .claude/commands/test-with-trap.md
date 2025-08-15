---
description: Run tests with bash trap cleanup mechanism  
allowed-tools: Bash(bash *), Bash(python test*), Bash(npx kill-port:*), Bash(ps:*), Bash(pkill:*)
---

## Test with Trap
Execute tests with trap-based process cleanup:
!`bash $CLAUDE_PROJECT_DIR/scripts/run-test.sh quick`

For complete RAG tests:
!`bash $CLAUDE_PROJECT_DIR/scripts/run-test.sh complete`

For full integration tests:
!`bash $CLAUDE_PROJECT_DIR/scripts/run-test.sh integration`
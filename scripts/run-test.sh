#!/usr/bin/env bash
# MCP Academic RAG Server 测试执行脚本
# 带有 trap 的一键回收机制

set -euo pipefail

# 设置 trap 在脚本退出时杀掉所有子进程
trap 'pkill -P $$ || true' EXIT

# 预清理
"$CLAUDE_PROJECT_DIR/.claude/hooks/cleanup.sh" || true

# 根据参数选择测试类型
case "${1:-quick}" in
    "complete"|"full")
        echo "🧪 运行完整RAG功能测试..."
        python test_rag_complete.py
        ;;
    "integration")
        echo "🧪 运行完整集成测试..."
        python test_complete_integration.py
        ;;
    "quick"|*)
        echo "🧪 运行快速API测试..."
        python test_quick_real_api.py
        ;;
esac

# 后清理
"$CLAUDE_PROJECT_DIR/.claude/hooks/cleanup.sh" || true

echo "✅ 测试完成，环境已清理"
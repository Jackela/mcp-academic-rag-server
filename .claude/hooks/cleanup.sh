#!/usr/bin/env bash
# Claude Code 项目清理钩子
# 关闭常见测试端口和项目相关子进程，避免残留

# Windows 环境适配
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    # Windows: 清理常见测试端口
    for port in 3000 3001 5173 9229 27017 6379 5000 8000 8080; do
        netstat -ano | findstr ":$port " | awk '{print $5}' | xargs -r taskkill //F //PID 2>/dev/null || true
    done
    
    # 清理项目相关的Python测试进程（但保留MCP Inspector等重要进程）
    tasklist //FI "IMAGENAME eq python.exe" | findstr "test" | awk '{print $2}' | xargs -r taskkill //F //PID 2>/dev/null || true
else
    # Unix-like: 清理常见端口
    npx --yes kill-port 3000 3001 5173 9229 27017 6379 5000 8000 8080 >/dev/null 2>&1 || true
    
    # 杀掉当前项目目录下的测试相关子进程
    pkill -f "python.*test.*$CLAUDE_PROJECT_DIR" >/dev/null 2>&1 || true
    pkill -f "node.*test.*$CLAUDE_PROJECT_DIR" >/dev/null 2>&1 || true
fi

# 清理临时文件
find "${CLAUDE_PROJECT_DIR:-$(pwd)}" -name "*.tmp" -delete 2>/dev/null || true
find "${CLAUDE_PROJECT_DIR:-$(pwd)}" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
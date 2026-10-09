#!/usr/bin/env python3
"""
发布MCP Academic RAG Server到PyPI的自动化脚本
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional, Sequence


def run_command(command: Sequence[str], description: str) -> Optional[str]:
    """运行命令并处理错误"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(list(command), check=True, capture_output=True, text=True)
        print(f"✅ {description}完成")
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"❌ {description}失败: {e.stderr}")
        return None


def main() -> bool:
    """主发布流程"""
    print("🚀 开始发布MCP Academic RAG Server到PyPI...")

    # 确保在项目根目录
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    print(f"📁 当前目录: {project_root}")

    # 步骤1: 清理之前的构建
    for directory in [project_root / "dist", project_root / "build", *project_root.glob("*.egg-info")]:
        if directory.is_dir() and not directory.is_symlink():
            shutil.rmtree(directory)

    # 步骤2: 构建包
    if run_command([sys.executable, "-m", "build"], "构建Python包") is None:
        print("❌ 构建失败，退出")
        return False

    # 步骤3: 检查包
    artifacts = sorted(str(path) for path in (project_root / "dist").iterdir() if path.is_file())
    if not artifacts or run_command([sys.executable, "-m", "twine", "check", *artifacts], "检查构建包") is None:
        print("❌ 包检查失败，退出")
        return False

    # 步骤4: 上传到PyPI
    print("\n📤 准备上传到PyPI...")
    print("请确保已设置PyPI凭据:")
    print("  - 方法1: pip install keyring, 然后 keyring set https://upload.pypi.org/legacy/ your-username")
    print("  - 方法2: 创建 ~/.pypirc 文件")
    print("  - 方法3: 使用 API token")

    confirm = input("\n✓ 确认上传到PyPI? (y/N): ")
    if confirm.lower() == "y":
        if run_command([sys.executable, "-m", "twine", "upload", *artifacts], "上传到PyPI") is not None:
            print("\n🎉 发布成功！")
            print("\n📋 现在用户可以一键安装:")
            print("  uvx mcp-academic-rag-server")
            print("\n🔧 Claude Desktop配置:")
            print("""  {
    "mcpServers": {
      "academic-rag": {
        "command": "uvx",
        "args": ["mcp-academic-rag-server"],
        "env": {
          "OPENAI_API_KEY": "sk-your-api-key-here"
        }
      }
    }
  }""")
            return True
    else:
        print("❌ 用户取消上传")
    return False


if __name__ == "__main__":
    # 检查依赖
    required_packages = ["build", "twine"]
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            print(f"❌ 缺少依赖: {package}")
            print(f"请运行: pip install {package}")
            sys.exit(1)

    main()

#!/usr/bin/env python3
"""
发布MCP Academic RAG Server到PyPI的自动化脚本
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """运行命令并处理错误"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description}完成")
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"❌ {description}失败: {e.stderr}")
        return None

def main():
    """主发布流程"""
    print("🚀 开始发布MCP Academic RAG Server到PyPI...")
    
    # 确保在项目根目录
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    print(f"📁 当前目录: {project_root}")
    
    # 步骤1: 清理之前的构建
    run_command("rm -rf dist/ build/ *.egg-info/", "清理构建目录")
    
    # 步骤2: 构建包
    if not run_command("python -m build", "构建Python包"):
        print("❌ 构建失败，退出")
        return False
    
    # 步骤3: 检查包
    if not run_command("python -m twine check dist/*", "检查构建包"):
        print("⚠️ 包检查有问题，但继续进行")
    
    # 步骤4: 上传到PyPI
    print("\n📤 准备上传到PyPI...")
    print("请确保已设置PyPI凭据:")
    print("  - 方法1: pip install keyring, 然后 keyring set https://upload.pypi.org/legacy/ your-username")
    print("  - 方法2: 创建 ~/.pypirc 文件")
    print("  - 方法3: 使用 API token")
    
    confirm = input("\n✓ 确认上传到PyPI? (y/N): ")
    if confirm.lower() == 'y':
        if run_command("python -m twine upload dist/*", "上传到PyPI"):
            print("\n🎉 发布成功！")
            print("\n📋 现在用户可以一键安装:")
            print("  uvx mcp-academic-rag-server")
            print("\n🔧 Claude Desktop配置:")
            print('''  {
    "mcpServers": {
      "academic-rag": {
        "command": "uvx",
        "args": ["mcp-academic-rag-server"],
        "env": {
          "OPENAI_API_KEY": "sk-your-api-key-here"
        }
      }
    }
  }''')
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
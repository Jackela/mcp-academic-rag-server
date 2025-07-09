"""
文档处理示例脚本

该脚本演示了如何使用文档处理CLI进行常见操作，包括：
- 上传和处理文档
- 查询文档信息
- 列出已处理文档
- 导出处理结果

用法：
    python document_processing_example.py [config_path]
"""

import os
import sys
import time
import argparse
from pathlib import Path

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cli.document_cli import DocumentCLI


def run_examples(config_path):
    """运行文档处理示例"""
    print("\n" + "=" * 80)
    print("文档处理示例")
    print("=" * 80 + "\n")
    
    # 创建CLI实例
    cli = DocumentCLI(config_path=config_path, verbose=True)
    
    # 示例目录
    example_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 创建示例文档
    example_doc_path = os.path.join(example_dir, "example_document.txt")
    with open(example_doc_path, 'w', encoding='utf-8') as f:
        f.write("这是一个示例文档，用于演示文档处理CLI的功能。\n")
        f.write("它包含简单的文本内容，适合进行处理流程测试。\n")
        f.write("\n示例章节 1\n")
        f.write("这是示例文档的第一个章节内容。\n")
        f.write("\n示例章节 2\n")
        f.write("这是示例文档的第二个章节内容。\n")
    
    print(f"已创建示例文档: {example_doc_path}")
    print("-" * 80 + "\n")
    
    # 示例1：上传并处理文档
    print("示例1：上传并处理文档")
    print("-" * 40)
    
    # 模拟命令行参数
    sys.argv = ["document_cli.py", "upload", "--file", example_doc_path]
    
    # 运行CLI
    cli.run()
    
    print("\n完成文档上传和处理")
    print("-" * 80 + "\n")
    time.sleep(1)  # 暂停，便于观察输出
    
    # 示例2：列出已处理文档
    print("示例2：列出已处理文档")
    print("-" * 40)
    
    # 模拟命令行参数
    sys.argv = ["document_cli.py", "list"]
    
    # 运行CLI
    cli.run()
    
    print("-" * 80 + "\n")
    time.sleep(1)  # 暂停，便于观察输出
    
    # 获取第一个文档ID（用于后续示例）
    # 在实际应用中，这里应该使用实际处理后的文档ID
    # 由于文档ID是动态生成的，这里我们需要从存储目录中获取
    storage_path = cli.storage_base_path
    document_dirs = [d for d in os.listdir(storage_path) if os.path.isdir(os.path.join(storage_path, d))]
    
    if not document_dirs:
        print("未找到已处理的文档，无法继续演示")
        return
    
    document_id = document_dirs[0]
    print(f"使用文档ID: {document_id} 进行后续操作\n")
    
    # 示例3：查询文档信息
    print("示例3：查询文档信息")
    print("-" * 40)
    
    # 模拟命令行参数
    sys.argv = ["document_cli.py", "info", "--id", document_id]
    
    # 运行CLI
    cli.run()
    
    print("-" * 80 + "\n")
    time.sleep(1)  # 暂停，便于观察输出
    
    # 示例4：导出文档
    print("示例4：导出文档")
    print("-" * 40)
    
    export_path = os.path.join(example_dir, "exported_document.md")
    
    # 模拟命令行参数
    sys.argv = ["document_cli.py", "export", "--id", document_id, 
                "--format", "markdown", "--output", export_path]
    
    # 运行CLI
    cli.run()
    
    print("\n导出的文档内容:")
    if os.path.exists(export_path):
        with open(export_path, 'r', encoding='utf-8') as f:
            print(f.read())
    
    print("-" * 80 + "\n")
    
    # 示例5：删除文档（添加确认以避免实际删除）
    print("示例5：删除文档")
    print("-" * 40)
    print("注意：实际运行时会提示确认，此示例不执行实际删除操作")
    
    # 模拟命令行参数（实际使用时应添加--confirm参数自动确认）
    # sys.argv = ["document_cli.py", "delete", "--id", document_id, "--confirm"]
    
    print("\n删除操作已跳过")
    print("-" * 80 + "\n")
    
    print("所有示例已完成运行")
    print("=" * 80)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="文档处理示例脚本")
    parser.add_argument("config_path", nargs="?", default="./config/config.json", 
                        help="配置文件路径")
    args = parser.parse_args()
    
    run_examples(args.config_path)


if __name__ == "__main__":
    main()

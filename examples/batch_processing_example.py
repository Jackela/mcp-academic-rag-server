"""
批量文档处理示例脚本

该脚本演示了如何批量处理目录中的多个文档，并展示处理性能指标。

用法：
    python batch_processing_example.py [config_path] [documents_dir]
"""

import os
import sys
import time
import argparse
import random
import string
from pathlib import Path

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cli.document_cli import DocumentCLI


def generate_test_documents(directory, count=10, size_range=(1, 5)):
    """
    生成测试文档
    
    Args:
        directory: 目标目录
        count: 文档数量
        size_range: 文档大小范围（KB）
        
    Returns:
        生成的文档路径列表
    """
    # 确保目录存在
    os.makedirs(directory, exist_ok=True)
    
    # 生成随机文本内容的函数
    def generate_random_text(size_kb):
        # 每KB大约1000个字符
        chars_count = size_kb * 1000
        # 生成随机文本
        paragraphs = []
        remaining_chars = chars_count
        
        while remaining_chars > 0:
            # 随机段落长度(200-500字符)
            paragraph_length = min(random.randint(200, 500), remaining_chars)
            
            # 生成随机段落
            paragraph = ''.join(random.choice(string.ascii_letters + string.digits + ' ' * 10) 
                               for _ in range(paragraph_length))
            
            paragraphs.append(paragraph)
            remaining_chars -= paragraph_length
        
        return '\n\n'.join(paragraphs)
    
    # 生成文档
    document_paths = []
    
    for i in range(count):
        # 随机文档大小
        size_kb = random.uniform(size_range[0], size_range[1])
        
        # 生成文件名
        file_name = f"test_doc_{i+1:03d}.txt"
        file_path = os.path.join(directory, file_name)
        
        # 生成内容
        content = generate_random_text(size_kb)
        
        # 写入文件
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        document_paths.append(file_path)
        
    return document_paths


def run_batch_processing(config_path, documents_dir=None):
    """运行批量处理示例"""
    print("\n" + "=" * 80)
    print("批量文档处理示例")
    print("=" * 80 + "\n")
    
    # 创建CLI实例
    cli = DocumentCLI(config_path=config_path)
    
    # 如果未指定文档目录，则在示例目录创建测试文档
    if not documents_dir:
        example_dir = os.path.dirname(os.path.abspath(__file__))
        documents_dir = os.path.join(example_dir, "test_documents")
    
    # 确保目录存在
    os.makedirs(documents_dir, exist_ok=True)
    
    # 检查目录中是否已有文档
    existing_docs = [f for f in os.listdir(documents_dir) 
                    if os.path.isfile(os.path.join(documents_dir, f)) 
                    and f.endswith('.txt')]
    
    # 如果目录为空，生成测试文档
    if not existing_docs:
        print(f"在 {documents_dir} 中生成测试文档...")
        document_paths = generate_test_documents(documents_dir, count=20)
        print(f"已生成 {len(document_paths)} 个测试文档")
    else:
        print(f"使用 {documents_dir} 中的 {len(existing_docs)} 个现有文档")
    
    print("-" * 80 + "\n")
    time.sleep(1)  # 暂停，便于观察输出
    
    # 批量处理目录中的文档
    print(f"开始批量处理 {documents_dir} 中的文档...")
    print("-" * 40)
    
    # 记录开始时间
    start_time = time.time()
    
    # 模拟命令行参数
    sys.argv = ["document_cli.py", "upload", "--directory", documents_dir,
               "--extensions", "txt", "--recursive"]
    
    # 运行CLI
    cli.run()
    
    # 计算总处理时间
    total_time = time.time() - start_time
    
    # 获取处理的文档数量
    processed_count = len([f for f in os.listdir(documents_dir) 
                          if os.path.isfile(os.path.join(documents_dir, f)) 
                          and f.endswith('.txt')])
    
    print("\n批量处理完成")
    print("-" * 80 + "\n")
    
    # 显示性能指标
    print("性能指标:")
    print(f"处理文档数量: {processed_count}")
    print(f"总处理时间: {total_time:.2f} 秒")
    print(f"平均每个文档处理时间: {total_time / processed_count:.4f} 秒")
    print(f"每秒处理文档数: {processed_count / total_time:.2f}")
    
    print("-" * 80 + "\n")
    
    # 列出处理后的文档
    print("处理后的文档列表:")
    print("-" * 40)
    
    # 模拟命令行参数
    sys.argv = ["document_cli.py", "list"]
    
    # 运行CLI
    cli.run()
    
    print("\n所有操作已完成")
    print("=" * 80)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="批量文档处理示例脚本")
    parser.add_argument("config_path", nargs="?", default="./config/config.json", 
                        help="配置文件路径")
    parser.add_argument("documents_dir", nargs="?", default=None,
                        help="要处理的文档目录，默认在示例目录创建测试文档")
    args = parser.parse_args()
    
    run_batch_processing(args.config_path, args.documents_dir)


if __name__ == "__main__":
    main()

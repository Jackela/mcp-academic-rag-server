"""
系统集成测试模块

该模块包含跨组件的集成测试，验证系统各部分的协同工作。
"""

import os
import sys
import unittest
import tempfile
import shutil
import json
import subprocess
from unittest.mock import patch, MagicMock

# 添加项目根目录到系统路径，确保能够导入被测试模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cli.document_cli import DocumentCLI
from cli.chat_cli import ChatCLI
from models.document import Document
from rag.chat_session import ChatSession


class TestSystemIntegration(unittest.TestCase):
    """系统集成测试类"""
    
    def setUp(self):
        """测试前准备"""
        # 创建临时目录
        self.temp_dir = tempfile.mkdtemp()
        
        # 创建临时配置文件
        self.config_path = os.path.join(self.temp_dir, "config.json")
        self.data_dir = os.path.join(self.temp_dir, "data")
        self.output_dir = os.path.join(self.temp_dir, "output")
        self.log_dir = os.path.join(self.temp_dir, "logs")
        
        # 创建配置内容
        config = {
            "storage": {
                "base_path": self.data_dir,
                "output_path": self.output_dir
            },
            "logging": {
                "level": "DEBUG",
                "file": os.path.join(self.log_dir, "test.log")
            }
        }
        
        # 写入配置文件
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)
        
        # 创建必要的目录
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(os.path.join(self.data_dir, "sessions"), exist_ok=True)
        
        # 创建测试文档
        self.test_file = os.path.join(self.temp_dir, "test_document.txt")
        with open(self.test_file, 'w', encoding='utf-8') as f:
            f.write("This is a test document for integration testing.")
    
    def tearDown(self):
        """测试后清理"""
        # 删除临时目录
        shutil.rmtree(self.temp_dir)
    
    def test_document_to_chat_flow(self):
        """测试从文档处理到聊天对话的完整流程"""
        # 第1步：使用DocumentCLI上传文档
        with patch('cli.document_cli.DocumentCLI._parse_args') as mock_parse_args:
            # 设置mock参数
            mock_args = MagicMock()
            mock_args.command = "upload"
            mock_args.file = self.test_file
            mock_args.directory = None
            mock_args.recursive = False
            mock_args.extensions = None
            mock_args.config = self.config_path
            mock_args.verbose = True
            mock_parse_args.return_value = mock_args
            
            # 运行文档CLI
            doc_cli = DocumentCLI(config_path=self.config_path)
            doc_cli.run()
        
        # 获取文档ID（在实际测试中，这里需要从日志或返回值中获取）
        # 由于这里使用模拟，我们直接使用一个固定的ID
        document_id = "test-document-id"
        
        # 为测试创建一个文档目录和文件
        document_dir = os.path.join(self.data_dir, document_id)
        os.makedirs(document_dir, exist_ok=True)
        
        # 创建文档JSON
        document_data = {
            "document_id": document_id,
            "file_path": self.test_file,
            "file_name": os.path.basename(self.test_file),
            "file_type": ".txt",
            "creation_time": "2023-01-01T00:00:00",
            "modification_time": "2023-01-01T00:00:00",
            "status": "completed",
            "metadata": {},
            "tags": ["test"],
            "content_stages": ["ocr", "structure"],
            "processing_history": [
                {
                    "time": "2023-01-01T00:00:00",
                    "status": "new"
                },
                {
                    "time": "2023-01-01T00:00:01",
                    "status": "processing"
                },
                {
                    "time": "2023-01-01T00:00:02",
                    "status": "completed"
                }
            ]
        }
        
        document_file = os.path.join(document_dir, "document.json")
        with open(document_file, 'w', encoding='utf-8') as f:
            json.dump(document_data, f, indent=2)
        
        # 第2步：使用文档处理CLI查询文档信息
        with patch('cli.document_cli.DocumentCLI._parse_args') as mock_parse_args:
            # 设置mock参数
            mock_args = MagicMock()
            mock_args.command = "info"
            mock_args.id = document_id
            mock_args.config = self.config_path
            mock_args.verbose = True
            mock_parse_args.return_value = mock_args
            
            # 运行文档CLI
            with patch('builtins.print') as mock_print:
                doc_cli = DocumentCLI(config_path=self.config_path)
                doc_cli.run()
                
                # 验证是否输出了文档信息
                mock_print.assert_any_call(f"文档信息 - ID: {document_id}")
        
        # 第3步：创建和使用聊天会话
        with patch('cli.chat_cli.ChatCLI._parse_args') as mock_parse_args:
            # 设置mock参数
            mock_args = MagicMock()
            mock_args.list = False
            mock_args.export = None
            mock_args.session = None
            mock_args.replay = False
            mock_args.config = self.config_path
            mock_args.verbose = True
            mock_parse_args.return_value = mock_args
            
            # 模拟用户输入和响应
            with patch('builtins.input', side_effect=["测试问题", "exit"]):
                with patch('cli.chat_cli.ChatCLI._process_user_input') as mock_process:
                    # 运行聊天CLI
                    chat_cli = ChatCLI(config_path=self.config_path)
                    chat_cli.run()
                    
                    # 验证是否处理了用户输入
                    mock_process.assert_called_once_with("测试问题")
    
    @patch('cli.document_cli.DocumentCLI')
    @patch('cli.chat_cli.ChatCLI')
    def test_command_line_interface(self, mock_chat_cli, mock_document_cli):
        """测试命令行界面入口点"""
        # 测试文档CLI
        try:
            # 在实际测试中，这里应使用subprocess模块运行CLI脚本
            # 例如：subprocess.run([sys.executable, "-m", "cli.document_cli", "--help"], check=True)
            # 由于这是集成测试，我们直接模拟入口点函数
            
            # 模拟document_cli.main()函数
            from cli.document_cli import main as doc_main
            with patch('sys.argv', ["document_cli.py", "--help"]):
                with patch('builtins.print'):  # 捕获帮助消息输出
                    try:
                        doc_main()
                    except SystemExit:
                        pass  # 忽略sys.exit()
        
        except Exception as e:
            self.fail(f"文档CLI入口点运行失败: {str(e)}")
        
        # 测试聊天CLI
        try:
            # 模拟chat_cli.main()函数
            from cli.chat_cli import main as chat_main
            with patch('sys.argv', ["chat_cli.py", "--help"]):
                with patch('builtins.print'):  # 捕获帮助消息输出
                    try:
                        chat_main()
                    except SystemExit:
                        pass  # 忽略sys.exit()
        
        except Exception as e:
            self.fail(f"聊天CLI入口点运行失败: {str(e)}")


class TestPerformance(unittest.TestCase):
    """性能测试类"""
    
    def setUp(self):
        """测试前准备"""
        # 创建临时目录
        self.temp_dir = tempfile.mkdtemp()
        
        # 创建临时配置文件
        self.config_path = os.path.join(self.temp_dir, "config.json")
        self.data_dir = os.path.join(self.temp_dir, "data")
        self.output_dir = os.path.join(self.temp_dir, "output")
        
        # 创建配置内容
        config = {
            "storage": {
                "base_path": self.data_dir,
                "output_path": self.output_dir
            }
        }
        
        # 写入配置文件
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)
        
        # 创建必要的目录
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 创建测试文档
        self.test_docs = []
        for i in range(10):
            test_file = os.path.join(self.temp_dir, f"test_document_{i}.txt")
            with open(test_file, 'w', encoding='utf-8') as f:
                f.write(f"This is test document {i} for performance testing.")
            self.test_docs.append(test_file)
    
    def tearDown(self):
        """测试后清理"""
        # 删除临时目录
        shutil.rmtree(self.temp_dir)
    
    @patch('time.sleep')  # 避免模拟处理延迟
    def test_batch_document_processing(self, mock_sleep):
        """测试批量文档处理性能"""
        import time
        
        # 设置mock
        mock_sleep.return_value = None
        
        # 创建CLI实例
        cli = DocumentCLI(config_path=self.config_path)
        
        # 创建mock参数
        mock_args = MagicMock()
        mock_args.directory = self.temp_dir
        mock_args.file = None
        mock_args.recursive = False
        mock_args.extensions = "txt"
        
        # 测量处理时间
        start_time = time.time()
        
        # 处理文档
        with patch('builtins.print'):  # 捕获输出
            cli._handle_upload(mock_args)
        
        # 计算总时间
        total_time = time.time() - start_time
        
        # 输出性能指标
        print(f"\n批量处理 {len(self.test_docs)} 个文档的总时间: {total_time:.2f} 秒")
        print(f"平均每个文档处理时间: {total_time / len(self.test_docs):.2f} 秒")
        
        # 注意：这个测试在模拟环境中不会反映真实性能
        # 在实际应用中，可以设置一个合理的阈值来验证性能
    
    def test_chat_response_time(self):
        """测试聊天响应时间"""
        import time
        
        # 创建CLI实例
        cli = ChatCLI(config_path=self.config_path)
        
        # 创建模拟会话
        session = cli._create_mock_session()
        cli.session = session
        
        # 测试查询
        test_queries = [
            "什么是机器学习？",
            "深度学习与传统机器学习有什么区别？",
            "自然语言处理的主要挑战是什么？",
            "如何评估模型性能？",
            "请解释强化学习的原理"
        ]
        
        total_time = 0
        
        # 测量响应时间
        for query in test_queries:
            # 测量处理时间
            start_time = time.time()
            
            # 处理查询（使用模拟实现）
            with patch('builtins.print'):  # 捕获输出
                response = cli._generate_mock_response(query)
            
            # 计算查询时间
            query_time = time.time() - start_time
            total_time += query_time
            
            print(f"\n查询: '{query}'")
            print(f"响应时间: {query_time:.2f} 秒")
        
        # 输出性能指标
        print(f"\n总计查询数: {len(test_queries)}")
        print(f"总响应时间: {total_time:.2f} 秒")
        print(f"平均响应时间: {total_time / len(test_queries):.2f} 秒")
        
        # 注意：这个测试在模拟环境中不会反映真实性能
        # 在实际应用中，可以设置一个合理的阈值来验证性能


if __name__ == '__main__':
    unittest.main()

"""
文档处理CLI测试模块

该模块包含针对document_cli.py的单元测试和集成测试。
"""

import os
import sys
import unittest
import tempfile
import shutil
import json
from unittest.mock import patch, MagicMock

# 添加项目根目录到系统路径，确保能够导入被测试模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cli.document_cli import DocumentCLI
from models.document import Document
from models.process_result import ProcessResult


class TestDocumentCLI(unittest.TestCase):
    """文档处理CLI测试类"""
    
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
    
    def tearDown(self):
        """测试后清理"""
        # 删除临时目录
        shutil.rmtree(self.temp_dir)
    
    @patch('cli.document_cli.Pipeline')
    def test_init(self, mock_pipeline):
        """测试初始化"""
        # 创建CLI实例
        cli = DocumentCLI(config_path=self.config_path, verbose=True)
        
        # 验证是否正确初始化
        self.assertEqual(cli.config_path, self.config_path)
        self.assertTrue(cli.verbose)
        self.assertEqual(cli.storage_base_path, self.data_dir)
        self.assertEqual(cli.storage_output_path, self.output_dir)
    
    @patch('cli.document_cli.DocumentCLI._parse_args')
    @patch('cli.document_cli.DocumentCLI._handle_upload')
    def test_run_upload_command(self, mock_handle_upload, mock_parse_args):
        """测试运行upload命令"""
        # 设置mock参数
        mock_args = MagicMock()
        mock_args.command = "upload"
        mock_args.config = self.config_path
        mock_args.verbose = False
        mock_parse_args.return_value = mock_args
        
        # 运行CLI
        cli = DocumentCLI(config_path=self.config_path)
        cli.run()
        
        # 验证是否调用了正确的处理函数
        mock_handle_upload.assert_called_once_with(mock_args)
    
    @patch('cli.document_cli.DocumentCLI._parse_args')
    @patch('cli.document_cli.DocumentCLI._handle_list')
    def test_run_list_command(self, mock_handle_list, mock_parse_args):
        """测试运行list命令"""
        # 设置mock参数
        mock_args = MagicMock()
        mock_args.command = "list"
        mock_args.config = self.config_path
        mock_args.verbose = False
        mock_parse_args.return_value = mock_args
        
        # 运行CLI
        cli = DocumentCLI(config_path=self.config_path)
        cli.run()
        
        # 验证是否调用了正确的处理函数
        mock_handle_list.assert_called_once_with(mock_args)
    
    @patch('builtins.print')
    @patch('cli.document_cli.os.path.exists')
    @patch('builtins.open', new_callable=unittest.mock.mock_open, read_data='{"document_id": "test-id", "file_name": "test.pdf", "creation_time": "2023-01-01T00:00:00", "modification_time": "2023-01-01T00:00:00", "status": "completed", "metadata": {}, "tags": [], "processing_history": []}')
    def test_handle_info(self, mock_open, mock_exists, mock_print):
        """测试处理info命令"""
        # 设置mock
        mock_exists.return_value = True
        
        # 创建CLI实例
        cli = DocumentCLI(config_path=self.config_path)
        
        # 创建mock参数
        mock_args = MagicMock()
        mock_args.id = "test-id"
        
        # 调用处理函数
        cli._handle_info(mock_args)
        
        # 验证是否输出了文档信息
        mock_print.assert_any_call("文档ID: test-id")
        mock_print.assert_any_call("文件名: test.pdf")
        mock_print.assert_any_call("创建时间: 2023-01-01T00:00:00")
        mock_print.assert_any_call("修改时间: 2023-01-01T00:00:00")
        mock_print.assert_any_call("状态: completed")
        mock_print.assert_any_call("元数据: {}")
        mock_print.assert_any_call("标签: []")
        mock_print.assert_any_call("处理历史: []")
    
    @patch('cli.document_cli.os.listdir')
    @patch('cli.document_cli.os.path.isdir')
    @patch('cli.document_cli.os.path.exists')
    @patch('builtins.open', new_callable=unittest.mock.mock_open, read_data='{"document_id": "test-id", "file_name": "test.pdf", "creation_time": "2023-01-01T00:00:00", "modification_time": "2023-01-01T00:00:00", "status": "completed", "metadata": {}, "tags": [], "processing_history": []}')
    @patch('builtins.print')
    def test_handle_list(self, mock_print, mock_open, mock_exists, mock_isdir, mock_listdir):
        """测试处理list命令"""
        # 设置mock
        mock_listdir.return_value = ["test-id"]
        mock_isdir.return_value = True
        mock_exists.return_value = True
        
        # 创建CLI实例
        cli = DocumentCLI(config_path=self.config_path)
        
        # 创建mock参数
        mock_args = MagicMock()
        mock_args.status = None
        mock_args.tag = None
        mock_args.format = "table"
        
        # 调用处理函数
        cli._handle_list(mock_args)
        
        # 验证是否输出了文档列表
        mock_print.assert_any_call("文档ID                               | 文件名                 | 状态         | 创建时间                 | 标签                  ")
        mock_print.assert_any_call("test-id                              | test.pdf               | completed   | 2023-01-01T00:00:00     | []                    ")


if __name__ == '__main__':
    unittest.main()

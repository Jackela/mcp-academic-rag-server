"""
聊天对话CLI测试模块

该模块包含针对chat_cli.py的单元测试和集成测试。
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

from cli.chat_cli import ChatCLI
from rag.chat_session import ChatSession, Message


class TestChatCLI(unittest.TestCase):
    """聊天对话CLI测试类"""
    
    def setUp(self):
        """测试前准备"""
        # 创建临时目录
        self.temp_dir = tempfile.mkdtemp()
        
        # 创建临时配置文件
        self.config_path = os.path.join(self.temp_dir, "config.json")
        self.data_dir = os.path.join(self.temp_dir, "data")
        self.sessions_dir = os.path.join(self.data_dir, "sessions")
        self.log_dir = os.path.join(self.temp_dir, "logs")
        
        # 创建配置内容
        config = {
            "storage": {
                "base_path": self.data_dir
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
        os.makedirs(self.sessions_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        
        # 创建测试会话
        self.test_session_id = "test-session"
        self.test_session_dir = os.path.join(self.sessions_dir, self.test_session_id)
        os.makedirs(self.test_session_dir, exist_ok=True)
        
        # 创建测试会话数据
        session_data = {
            "session_id": self.test_session_id,
            "max_history_length": 10,
            "metadata": {},
            "messages": [
                {
                    "message_id": "msg1",
                    "role": "system",
                    "content": "这是一个测试系统消息",
                    "timestamp": 1672531200.0,
                    "metadata": {}
                },
                {
                    "message_id": "msg2",
                    "role": "user",
                    "content": "测试问题",
                    "timestamp": 1672531300.0,
                    "metadata": {}
                },
                {
                    "message_id": "msg3",
                    "role": "assistant",
                    "content": "测试回答",
                    "timestamp": 1672531400.0,
                    "metadata": {}
                }
            ],
            "citations": {},
            "created_at": 1672531200.0,
            "last_active_at": 1672531400.0
        }
        
        # 写入测试会话文件
        session_file = os.path.join(self.test_session_dir, "session.json")
        with open(session_file, 'w', encoding='utf-8') as f:
            json.dump(session_data, f, indent=2)
    
    def tearDown(self):
        """测试后清理"""
        # 删除临时目录
        shutil.rmtree(self.temp_dir)
    
    def test_init(self):
        """测试初始化"""
        # 创建CLI实例
        cli = ChatCLI(config_path=self.config_path, verbose=True)
        
        # 验证是否正确初始化
        self.assertEqual(cli.config_path, self.config_path)
        self.assertTrue(cli.verbose)
        self.assertEqual(cli.sessions_dir, self.sessions_dir)
    
    @patch('cli.chat_cli.ChatCLI._parse_args')
    @patch('cli.chat_cli.ChatCLI._list_sessions')
    def test_run_list_command(self, mock_list_sessions, mock_parse_args):
        """测试运行list命令"""
        # 设置mock参数
        mock_args = MagicMock()
        mock_args.list = True
        mock_args.export = None
        mock_args.session = None
        mock_args.replay = False
        mock_args.config = self.config_path
        mock_args.verbose = False
        mock_parse_args.return_value = mock_args
        
        # 运行CLI
        cli = ChatCLI(config_path=self.config_path)
        cli.run()
        
        # 验证是否调用了正确的处理函数
        mock_list_sessions.assert_called_once()
    
    @patch('cli.chat_cli.ChatCLI._parse_args')
    @patch('cli.chat_cli.ChatCLI._export_session')
    def test_run_export_command(self, mock_export_session, mock_parse_args):
        """测试运行export命令"""
        # 设置mock参数
        mock_args = MagicMock()
        mock_args.list = False
        mock_args.export = self.test_session_id
        mock_args.session = None
        mock_args.replay = False
        mock_args.config = self.config_path
        mock_args.verbose = False
        mock_parse_args.return_value = mock_args
        
        # 运行CLI
        cli = ChatCLI(config_path=self.config_path)
        cli.run()
        
        # 验证是否调用了正确的处理函数
        mock_export_session.assert_called_once_with(self.test_session_id)
    
    @patch('cli.chat_cli.ChatCLI._parse_args')
    @patch('cli.chat_cli.ChatCLI._load_session')
    @patch('cli.chat_cli.ChatCLI._replay_session')
    def test_run_replay_command(self, mock_replay_session, mock_load_session, mock_parse_args):
        """测试运行replay命令"""
        # 设置mock参数
        mock_args = MagicMock()
        mock_args.list = False
        mock_args.export = None
        mock_args.session = self.test_session_id
        mock_args.replay = True
        mock_args.config = self.config_path
        mock_args.verbose = False
        mock_parse_args.return_value = mock_args
        
        # 运行CLI
        cli = ChatCLI(config_path=self.config_path)
        cli.run()
        
        # 验证是否调用了正确的处理函数
        mock_load_session.assert_called_once_with(self.test_session_id)
        mock_replay_session.assert_called_once()
    
    @patch('builtins.print')
    @patch('os.listdir')
    @patch('os.path.isdir')
    @patch('os.path.exists')
    @patch('builtins.open', new_callable=unittest.mock.mock_open, read_data='{"session_id": "test-session", "created_at": 1672531200.0, "last_active_at": 1672531400.0, "messages": [{"role": "user", "content": "test"}]}')
    def test_list_sessions(self, mock_open, mock_exists, mock_isdir, mock_listdir, mock_print):
        """测试列出会话"""
        # 设置mock
        mock_listdir.return_value = [self.test_session_id]
        mock_isdir.return_value = True
        mock_exists.return_value = True
        
        # 创建CLI实例
        cli = ChatCLI(config_path=self.config_path)
        
        # 调用列出会话函数
        cli._list_sessions()
        
        # 验证是否输出了会话列表
        mock_print.assert_any_call("会话ID                               | 创建时间                 | 最后活动时间             | 消息数   ")
    
    @patch('builtins.open', new_callable=unittest.mock.mock_open)
    @patch('os.path.exists')
    def test_load_session(self, mock_exists, mock_open):
        """测试加载会话"""
        # 设置mock
        mock_exists.return_value = True
        mock_open.return_value.__enter__.return_value.read.return_value = json.dumps({
            "session_id": self.test_session_id,
            "max_history_length": 10,
            "metadata": {},
            "messages": [
                {
                    "message_id": "msg1",
                    "role": "system",
                    "content": "这是一个测试系统消息",
                    "timestamp": 1672531200.0,
                    "metadata": {}
                }
            ],
            "citations": {},
            "created_at": 1672531200.0,
            "last_active_at": 1672531400.0
        })
        
        # 创建CLI实例，使用patch避免实际读取文件
        with patch('rag.chat_session.ChatSession.from_dict') as mock_from_dict:
            mock_from_dict.return_value = MagicMock()
            
            cli = ChatCLI(config_path=self.config_path)
            cli._load_session(self.test_session_id)
            
            # 验证是否调用了from_dict方法
            mock_from_dict.assert_called_once()
    
    @patch('builtins.print')
    def test_format_timestamp(self, mock_print):
        """测试格式化时间戳"""
        # 创建CLI实例
        cli = ChatCLI(config_path=self.config_path)
        
        # 测试有效时间戳
        formatted = cli._format_timestamp(1672531200.0)
        self.assertEqual(formatted, "2023-01-01 08:00:00")
        
        # 测试无效时间戳
        formatted = cli._format_timestamp(None)
        self.assertEqual(formatted, "未知")
        
        formatted = cli._format_timestamp("invalid")
        self.assertEqual(formatted, "invalid")


if __name__ == '__main__':
    unittest.main()

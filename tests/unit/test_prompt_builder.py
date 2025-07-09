"""
提示构建器单元测试
"""

import pytest
from unittest.mock import patch, MagicMock
from haystack.dataclasses import Document as HaystackDocument, ChatMessage

from rag.prompt_builder import ChatPromptBuilder, PromptBuilderFactory


class TestChatPromptBuilder:
    """聊天提示构建器单元测试类"""
    
    @pytest.fixture
    def sample_documents(self):
        """样本文档"""
        return [
            HaystackDocument(
                content="This is the content of document 1", 
                id="doc1", 
                metadata={"title": "Document 1"}
            ),
            HaystackDocument(
                content="This is the content of document 2", 
                id="doc2", 
                metadata={"title": "Document 2"}
            )
        ]
    
    @pytest.fixture
    def sample_chat_history(self):
        """样本聊天历史"""
        return [
            {"role": "user", "content": "Previous question"},
            {"role": "assistant", "content": "Previous answer"}
        ]
    
    def test_initialization_default(self):
        """测试默认初始化"""
        builder = ChatPromptBuilder()
        
        # 验证默认值
        assert builder.document_separator == "\n---\n"
        assert builder.include_citation is True
        assert builder.max_context_length == 4000
        assert builder.template_type == "academic"
        assert builder.system_prompt is not None
    
    def test_initialization_custom(self):
        """测试自定义初始化"""
        custom_prompt = "Custom system prompt"
        builder = ChatPromptBuilder(
            system_prompt=custom_prompt,
            document_separator="===",
            include_citation=False,
            max_context_length=2000,
            template_type="concise"
        )
        
        # 验证自定义值
        assert builder.document_separator == "==="
        assert builder.include_citation is False
        assert builder.max_context_length == 2000
        assert builder.template_type == "concise"
        assert builder.system_prompt == custom_prompt
    
    def test_get_default_system_prompt(self):
        """测试获取默认系统提示"""
        # 测试学术模板
        builder = ChatPromptBuilder(template_type="academic")
        prompt = builder._get_default_system_prompt()
        assert "学术研究助手" in prompt
        assert "基于上下文信息回答问题" in prompt
        
        # 测试简洁模板
        builder = ChatPromptBuilder(template_type="concise")
        prompt = builder._get_default_system_prompt()
        assert "简洁的学术助手" in prompt
        assert "简明的回答" in prompt
        
        # 测试通用模板
        builder = ChatPromptBuilder(template_type="general")
        prompt = builder._get_default_system_prompt()
        assert "智能助手" in prompt
        assert "文档内容的问题" in prompt
    
    def test_format_documents(self, sample_documents):
        """测试格式化文档"""
        builder = ChatPromptBuilder(include_citation=True)
        formatted = builder._format_documents(sample_documents)
        
        # 验证格式
        assert "文档ID: doc1" in formatted
        assert "标题: Document 1" in formatted
        assert "This is the content of document 1" in formatted
        assert "文档ID: doc2" in formatted
        assert "标题: Document 2" in formatted
        assert "This is the content of document 2" in formatted
        
        # 测试没有引用的情况
        builder = ChatPromptBuilder(include_citation=False)
        formatted = builder._format_documents(sample_documents)
        
        # 验证格式
        assert "文档ID: doc1" not in formatted
        assert "标题: Document 1" not in formatted
        assert "This is the content of document 1" in formatted
        assert "This is the content of document 2" in formatted
    
    def test_format_documents_max_length(self):
        """测试格式化文档时的最大长度限制"""
        # 创建长文档
        long_docs = [
            HaystackDocument(
                content="A" * 2000, 
                id="doc1", 
                metadata={"title": "Long Document 1"}
            ),
            HaystackDocument(
                content="B" * 2000, 
                id="doc2", 
                metadata={"title": "Long Document 2"}
            ),
            HaystackDocument(
                content="C" * 2000, 
                id="doc3", 
                metadata={"title": "Long Document 3"}
            )
        ]
        
        # 设置较小的最大上下文长度
        builder = ChatPromptBuilder(max_context_length=1000)
        formatted = builder._format_documents(long_docs)
        
        # 验证长度限制
        assert len(formatted) <= 1000
        assert "Long Document 1" in formatted
        assert "Long Document 3" not in formatted  # 第三个文档应该被截断
    
    def test_build_academic_prompt(self):
        """测试构建学术提示"""
        builder = ChatPromptBuilder()
        prompt = builder._build_academic_prompt("Test query", "Test context")
        
        # 验证提示格式
        assert "请回答以下学术问题" in prompt
        assert "问题：Test query" in prompt
        assert "参考文献内容：\nTest context" in prompt
        assert "仅基于提供的参考文献内容回答" in prompt
    
    def test_build_concise_prompt(self):
        """测试构建简洁提示"""
        builder = ChatPromptBuilder()
        prompt = builder._build_concise_prompt("Test query", "Test context")
        
        # 验证提示格式
        assert "简明扼要地回答问题" in prompt
        assert "问题：Test query" in prompt
        assert "参考内容：\nTest context" in prompt
        assert "回答简洁准确" in prompt
    
    def test_build_general_prompt(self):
        """测试构建通用提示"""
        builder = ChatPromptBuilder()
        prompt = builder._build_general_prompt("Test query", "Test context")
        
        # 验证提示格式
        assert "根据以下文档内容回答问题" in prompt
        assert "问题：Test query" in prompt
        assert "参考文档：\nTest context" in prompt
        assert "基于参考文档提供准确" in prompt
    
    def test_call_method(self, sample_documents, sample_chat_history):
        """测试__call__方法"""
        builder = ChatPromptBuilder()
        
        # 模拟方法
        with patch.object(builder, '_format_documents') as mock_format, \
             patch.object(builder, '_build_academic_prompt') as mock_build_prompt:
            
            mock_format.return_value = "Formatted context"
            mock_build_prompt.return_value = "Built prompt"
            
            # 调用__call__方法
            result = builder(
                query="Test query",
                documents=sample_documents,
                chat_history=sample_chat_history
            )
            
            # 验证方法调用
            mock_format.assert_called_once_with(sample_documents)
            mock_build_prompt.assert_called_once_with("Test query", "Formatted context")
            
            # 验证结果
            assert "messages" in result
            messages = result["messages"]
            assert len(messages) > 0
            
            # 验证系统消息
            system_message = messages[0]
            assert isinstance(system_message, ChatMessage)
            assert system_message.role == "system"
            assert system_message.content == builder.system_prompt
            
            # 验证历史消息
            for i, history_item in enumerate(sample_chat_history):
                message = messages[i+1]
                assert message.role == history_item["role"]
                assert message.content == history_item["content"]
            
            # 验证用户消息
            user_message = messages[-1]
            assert user_message.role == "user"
            assert user_message.content == "Built prompt"


class TestPromptBuilderFactory:
    """提示构建器工厂单元测试类"""
    
    def test_create_builder_default(self):
        """测试创建默认提示构建器"""
        builder = PromptBuilderFactory.create_builder()
        
        assert isinstance(builder, ChatPromptBuilder)
        assert builder.template_type == "academic"
        assert builder.include_citation is True
        assert builder.max_context_length == 4000
    
    def test_create_builder_custom(self):
        """测试创建自定义提示构建器"""
        custom_prompt = "Custom system prompt"
        builder = PromptBuilderFactory.create_builder(
            template_type="concise",
            system_prompt=custom_prompt,
            include_citation=False,
            max_context_length=2000
        )
        
        assert isinstance(builder, ChatPromptBuilder)
        assert builder.template_type == "concise"
        assert builder.system_prompt == custom_prompt
        assert builder.include_citation is False
        assert builder.max_context_length == 2000
    
    def test_create_builder_with_config(self):
        """测试使用配置创建提示构建器"""
        config = {
            "template_type": "general",
            "include_citation": False,
            "max_context_length": 3000,
            "system_prompt": "Config system prompt"
        }
        
        builder = PromptBuilderFactory.create_builder(config=config)
        
        assert isinstance(builder, ChatPromptBuilder)
        assert builder.template_type == "general"
        assert builder.system_prompt == "Config system prompt"
        assert builder.include_citation is False
        assert builder.max_context_length == 3000

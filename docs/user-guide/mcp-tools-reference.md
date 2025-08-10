# MCP工具完整参考文档

## 概述

MCP Academic RAG Server 提供了一套完整的工具，让AI助手能够直接处理和查询学术文档。本文档详细描述了所有可用工具的参数、使用方法和最佳实践。

## 工具列表

| 工具名称 | 功能描述 | 状态 |
|---------|----------|------|
| [`process_document`](#process_document) | 处理学术文档 | ✅ 稳定 |
| [`query_documents`](#query_documents) | 查询已处理文档 | ✅ 稳定 |
| [`get_document_info`](#get_document_info) | 获取文档信息 | ✅ 稳定 |
| [`list_sessions`](#list_sessions) | 列出聊天会话 | ✅ 稳定 |

---

## process_document

处理学术文档，包括OCR、结构识别、分类和向量化。

### 参数

#### 必需参数

| 参数名 | 类型 | 描述 | 示例 |
|-------|------|------|------|
| `file_path` | string | 文档文件的完整路径 | `/home/user/papers/paper.pdf` |

#### 可选参数

| 参数名 | 类型 | 默认值 | 描述 | 示例 |
|-------|------|--------|------|------|
| `file_name` | string | 文件名 | 文档的显示名称 | `"机器学习综述.pdf"` |

### 输入格式

```json
{
  "file_path": "/path/to/document.pdf",
  "file_name": "研究论文.pdf"
}
```

### 输出格式

#### 成功响应

```json
{
  "status": "success",
  "document_id": "doc_abc123",
  "file_name": "研究论文.pdf",
  "processing_stages": [
    "PreProcessor",
    "OCRProcessor", 
    "StructureProcessor",
    "ClassificationProcessor",
    "EmbeddingProcessor"
  ],
  "metadata": {
    "page_count": 15,
    "language": "zh",
    "document_type": "academic_paper",
    "title": "深度学习在自然语言处理中的应用",
    "authors": ["张三", "李四"],
    "abstract": "本文综述了深度学习在NLP领域的最新进展...",
    "keywords": ["深度学习", "自然语言处理", "神经网络"],
    "processing_time": 45.2,
    "file_size": "2.5MB"
  },
  "message": "Document processed successfully"
}
```

#### 错误响应

```json
{
  "status": "error",
  "message": "File not found: /path/to/document.pdf",
  "error": "FileNotFoundError"
}
```

### 使用示例

#### 基本用法
```
请处理这篇论文：/home/user/research/deep_learning_survey.pdf
```

#### 指定文件名
```
处理文档 /docs/paper.pdf，显示名称为"深度学习综述论文"
```

#### 批量处理
```
请依次处理以下文档：
1. /papers/paper1.pdf
2. /papers/paper2.pdf  
3. /papers/paper3.pdf
```

### 支持的文件格式

| 格式 | 扩展名 | 处理方式 | 备注 |
|------|--------|----------|------|
| PDF | `.pdf` | OCR + 文本提取 | 推荐格式 |
| 图片 | `.jpg`, `.png`, `.tiff` | OCR识别 | 高分辨率效果更好 |
| Word | `.docx`, `.doc` | 直接文本提取 | 保留格式信息 |
| PowerPoint | `.pptx`, `.ppt` | 幻灯片内容提取 | 包含图片文字 |

### 处理阶段说明

1. **PreProcessor**: 图像预处理，提升OCR质量
2. **OCRProcessor**: 光学字符识别，提取文本内容
3. **StructureProcessor**: 结构识别，提取标题、段落、表格等
4. **ClassificationProcessor**: 内容分类，识别文档类型和主题
5. **EmbeddingProcessor**: 向量化，生成语义向量表示

### 错误码说明

| 错误码 | 描述 | 解决方法 |
|--------|------|----------|
| `FileNotFoundError` | 文件不存在 | 检查文件路径是否正确 |
| `UnsupportedFormat` | 不支持的文件格式 | 使用支持的文件格式 |
| `FileTooLarge` | 文件过大 | 文件大小应小于16MB |
| `OCRError` | OCR处理失败 | 检查图像质量，重试处理 |
| `ProcessingTimeout` | 处理超时 | 大文件可能需要更长时间 |

---

## query_documents

基于已处理文档回答问题，支持上下文对话。

### 参数

#### 必需参数

| 参数名 | 类型 | 描述 | 示例 |
|-------|------|------|------|
| `query` | string | 要查询的问题或话题 | `"什么是深度学习？"` |

#### 可选参数

| 参数名 | 类型 | 默认值 | 值域 | 描述 |
|-------|------|--------|------|------|
| `session_id` | string | 自动生成 | - | 会话ID，用于保持对话上下文 |
| `top_k` | integer | 5 | 1-20 | 返回的相关文档数量 |

### 输入格式

```json
{
  "query": "深度学习和传统机器学习有什么区别？",
  "session_id": "session_xyz789",
  "top_k": 5
}
```

### 输出格式

#### 成功响应

```json
{
  "status": "success",
  "session_id": "session_xyz789",
  "query": "深度学习和传统机器学习有什么区别？",
  "answer": "根据文献，深度学习与传统机器学习的主要区别包括：\n\n1. **特征工程**：传统机器学习需要手动设计特征，而深度学习能够自动学习特征表示...\n\n2. **模型复杂度**：深度学习使用多层神经网络，能够建模更复杂的非线性关系...\n\n3. **数据需求**：深度学习通常需要大量数据才能达到最佳性能...",
  "sources": [
    {
      "content": "深度学习是机器学习的一个子领域，使用具有多个隐藏层的神经网络来建模和理解复杂的模式...",
      "metadata": {
        "title": "深度学习综述",
        "authors": ["张三", "李四"],
        "page": 3,
        "document_id": "doc_abc123",
        "confidence": 0.95,
        "relevance_score": 0.89
      },
      "structured_content": {
        "type": "table",
        "title": "深度学习vs传统机器学习对比",
        "data": {
          "headers": ["特征", "传统ML", "深度学习"],
          "rows": [
            ["特征工程", "手动", "自动"],
            ["数据需求", "中等", "大量"],
            ["解释性", "高", "低"]
          ]
        }
      }
    }
  ],
  "query_metadata": {
    "processing_time": 1.2,
    "retrieval_method": "hybrid",
    "llm_model": "gpt-3.5-turbo",
    "total_documents_searched": 156
  }
}
```

### 查询类型和技巧

#### 1. 事实性查询
```
什么是Transformer架构？
深度学习有哪些主要应用领域？
BERT模型是如何工作的？
```

#### 2. 比较分析
```
比较CNN和RNN的优缺点
Transformer与RNN在序列建模上有什么区别？
监督学习和无监督学习的主要差异是什么？
```

#### 3. 综述性问题
```
总结一下计算机视觉领域的最新进展
深度学习在自然语言处理中有哪些突破？
强化学习的发展历程是怎样的？
```

#### 4. 技术实现
```
如何实现一个简单的神经网络？
训练深度学习模型需要注意什么？
如何处理过拟合问题？
```

#### 5. 上下文对话
```
# 第一轮
什么是注意力机制？

# 第二轮（基于上下文）
它是如何在Transformer中应用的？

# 第三轮（继续上下文）
相比传统的RNN，它有什么优势？
```

### 高级查询功能

#### 过滤和限制
```json
{
  "query": "深度学习应用",
  "filters": {
    "document_type": "academic_paper",
    "language": "en",
    "year_range": [2020, 2024]
  },
  "top_k": 10
}
```

#### 指定检索策略
```json
{
  "query": "神经网络优化方法",
  "retrieval_config": {
    "method": "hybrid",
    "dense_weight": 0.7,
    "sparse_weight": 0.3
  }
}
```

---

## get_document_info

获取指定文档的详细处理信息和元数据。

### 参数

#### 必需参数

| 参数名 | 类型 | 描述 | 示例 |
|-------|------|------|------|
| `document_id` | string | 文档的唯一标识符 | `"doc_abc123"` |

### 输入格式

```json
{
  "document_id": "doc_abc123"
}
```

### 输出格式

```json
{
  "status": "success",
  "document_info": {
    "id": "doc_abc123",
    "file_name": "深度学习综述.pdf",
    "file_path": "/processed/docs/doc_abc123.pdf",
    "file_size": "2.5MB",
    "upload_time": "2024-01-15T10:30:00Z",
    "processing_time": 45.2,
    "processing_status": "completed",
    "metadata": {
      "title": "深度学习在自然语言处理中的应用",
      "authors": ["张三", "李四", "王五"],
      "abstract": "本文全面综述了深度学习技术在自然语言处理领域的应用现状、挑战和未来发展方向...",
      "keywords": ["深度学习", "自然语言处理", "神经网络", "Transformer"],
      "language": "zh",
      "page_count": 15,
      "document_type": "academic_paper",
      "subject_area": "计算机科学",
      "publication_year": 2023,
      "doi": "10.1000/182",
      "citations_count": 156
    },
    "structure": {
      "sections": [
        {"title": "摘要", "page_range": [1, 1], "word_count": 245},
        {"title": "引言", "page_range": [2, 3], "word_count": 892},
        {"title": "相关工作", "page_range": [4, 6], "word_count": 1456},
        {"title": "方法", "page_range": [7, 11], "word_count": 2134},
        {"title": "实验", "page_range": [12, 14], "word_count": 1678},
        {"title": "结论", "page_range": [15, 15], "word_count": 423}
      ],
      "tables": [
        {"title": "实验结果对比", "page": 13, "row_count": 8, "col_count": 5},
        {"title": "模型参数统计", "page": 14, "row_count": 6, "col_count": 3}
      ],
      "figures": [
        {"title": "模型架构图", "page": 8, "type": "diagram"},
        {"title": "训练损失曲线", "page": 12, "type": "chart"},
        {"title": "注意力热力图", "page": 13, "type": "heatmap"}
      ]
    },
    "processing_stages": {
      "PreProcessor": {
        "status": "completed",
        "duration": 2.1,
        "output": "图像预处理完成，提升对比度"
      },
      "OCRProcessor": {
        "status": "completed", 
        "duration": 28.5,
        "output": "提取文本15,847字符，置信度98.2%"
      },
      "StructureProcessor": {
        "status": "completed",
        "duration": 8.3,
        "output": "识别6个章节，2个表格，3个图表"
      },
      "ClassificationProcessor": {
        "status": "completed",
        "duration": 3.1,
        "output": "分类：学术论文，置信度95.7%"
      },
      "EmbeddingProcessor": {
        "status": "completed",
        "duration": 3.2,
        "output": "生成384维向量表示"
      }
    },
    "quality_metrics": {
      "ocr_confidence": 0.982,
      "structure_completeness": 0.945,
      "classification_confidence": 0.957,
      "embedding_quality": 0.891
    }
  }
}
```

### 使用示例

```
显示文档 doc_abc123 的详细信息
请提供文档ID为 doc_xyz789 的处理状态
查看最近处理的文档的元数据信息
```

---

## list_sessions

列出所有聊天会话及其基本信息。

### 参数

无需参数。

### 输入格式

```json
{}
```

### 输出格式

```json
{
  "status": "success",
  "sessions": [
    {
      "session_id": "session_abc123",
      "created_at": "2024-01-15T09:00:00Z",
      "last_active_at": "2024-01-15T10:30:00Z",
      "message_count": 12,
      "metadata": {
        "user_id": "user_001",
        "topic": "深度学习研究",
        "document_count": 5,
        "avg_response_time": 1.8
      }
    },
    {
      "session_id": "session_xyz789", 
      "created_at": "2024-01-14T14:20:00Z",
      "last_active_at": "2024-01-14T15:45:00Z",
      "message_count": 8,
      "metadata": {
        "user_id": "user_002",
        "topic": "自然语言处理",
        "document_count": 3,
        "avg_response_time": 2.1
      }
    }
  ],
  "total_count": 2,
  "active_sessions": 1,
  "statistics": {
    "total_messages": 20,
    "avg_messages_per_session": 10,
    "most_active_session": "session_abc123",
    "oldest_session": "2024-01-10T08:00:00Z"
  }
}
```

### 使用示例

```
显示所有聊天会话
列出我的对话历史
查看会话统计信息
```

---

## 通用错误处理

### 标准错误格式

```json
{
  "status": "error",
  "error_code": "INVALID_PARAMETER",
  "message": "Parameter 'file_path' is required but was not provided",
  "details": {
    "parameter": "file_path",
    "expected_type": "string",
    "received": null
  },
  "timestamp": "2024-01-15T10:30:00Z",
  "request_id": "req_abc123"
}
```

### 常见错误码

| 错误码 | 描述 | HTTP状态码 | 解决方法 |
|--------|------|------------|----------|
| `INVALID_PARAMETER` | 参数无效或缺失 | 400 | 检查参数格式和必填项 |
| `FILE_NOT_FOUND` | 文件不存在 | 404 | 确认文件路径正确 |
| `UNSUPPORTED_FORMAT` | 不支持的文件格式 | 400 | 使用支持的文件格式 |
| `PROCESSING_ERROR` | 处理过程出错 | 500 | 重试或联系支持 |
| `RATE_LIMIT_EXCEEDED` | 请求频率过高 | 429 | 降低请求频率 |
| `AUTHENTICATION_FAILED` | 认证失败 | 401 | 检查API密钥 |
| `INSUFFICIENT_STORAGE` | 存储空间不足 | 507 | 清理旧文件或扩容 |

---

## 最佳实践

### 1. 会话管理

**推荐做法**:
- 为每个用户或任务创建独立会话
- 在相关主题的多轮对话中保持同一会话ID
- 定期清理不活跃的会话

**示例**:
```
# 开始新主题
query_documents(query="什么是机器学习？")  # 新会话

# 继续同一主题
query_documents(query="它有哪些应用？", session_id="session_123")
query_documents(query="与深度学习的区别？", session_id="session_123")
```

### 2. 文档处理优化

**文件准备**:
- 使用高分辨率扫描（300 DPI以上）
- 确保文字清晰，避免模糊和倾斜
- PDF格式优于图片格式
- 文件大小控制在16MB以内

**批量处理**:
```python
# 推荐的批量处理方式
documents = [
    "/papers/paper1.pdf",
    "/papers/paper2.pdf", 
    "/papers/paper3.pdf"
]

for doc_path in documents:
    result = process_document(file_path=doc_path)
    if result.status == "success":
        print(f"✅ {doc_path} 处理完成")
    else:
        print(f"❌ {doc_path} 处理失败: {result.message}")
```

### 3. 查询优化技巧

**具体化查询**:
```
# 好的查询
"Transformer架构中自注意力机制的计算复杂度是多少？"

# 避免过于宽泛的查询  
"机器学习"
```

**结构化查询**:
```
"请对比以下三种优化算法的优缺点：
1. SGD (随机梯度下降)
2. Adam
3. RMSprop

并生成对比表格"
```

**上下文利用**:
```
# 第一轮：建立上下文
"解释一下BERT模型的架构"

# 第二轮：基于上下文深入
"它相比GPT有什么创新之处？"

# 第三轮：实际应用
"在情感分析任务中如何使用它？"
```

### 4. 错误处理和重试

**重试策略**:
```python
import time
import random

def process_with_retry(file_path, max_retries=3):
    for attempt in range(max_retries):
        try:
            result = process_document(file_path=file_path)
            if result.status == "success":
                return result
        except Exception as e:
            if attempt < max_retries - 1:
                # 指数退避
                wait_time = (2 ** attempt) + random.uniform(0, 1)
                time.sleep(wait_time)
            else:
                raise e
```

**错误监控**:
```python
def monitor_processing(file_path):
    try:
        result = process_document(file_path=file_path)
        
        # 检查处理质量
        if result.metadata.get("ocr_confidence", 0) < 0.8:
            print("⚠️ OCR置信度较低，建议检查原文档质量")
            
        return result
    except Exception as e:
        print(f"❌ 处理失败: {e}")
        return None
```

### 5. 性能优化

**并发处理**:
```python
import asyncio

async def process_documents_concurrently(file_paths):
    tasks = []
    for file_path in file_paths:
        task = asyncio.create_task(
            process_document(file_path=file_path)
        )
        tasks.append(task)
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return results
```

**缓存查询结果**:
```python
from functools import lru_cache

@lru_cache(maxsize=100)
def cached_query(query_text, session_id=None):
    return query_documents(query=query_text, session_id=session_id)
```

---

## 集成示例

### Python集成

```python
import json
import requests

class MCPAcademicRAGClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        
    def process_document(self, file_path, file_name=None):
        """处理文档"""
        data = {"file_path": file_path}
        if file_name:
            data["file_name"] = file_name
            
        response = requests.post(
            f"{self.base_url}/mcp/process_document",
            json=data
        )
        return response.json()
    
    def query(self, query, session_id=None, top_k=5):
        """查询文档"""
        data = {
            "query": query,
            "top_k": top_k
        }
        if session_id:
            data["session_id"] = session_id
            
        response = requests.post(
            f"{self.base_url}/mcp/query_documents", 
            json=data
        )
        return response.json()

# 使用示例
client = MCPAcademicRAGClient()

# 处理文档
result = client.process_document("/path/to/paper.pdf")
print(f"文档ID: {result['document_id']}")

# 查询
answer = client.query("什么是深度学习？")
print(f"回答: {answer['answer']}")
```

### cURL集成

```bash
#!/bin/bash

# 处理文档
curl -X POST http://localhost:8000/mcp/process_document \
  -H "Content-Type: application/json" \
  -d '{
    "file_path": "/home/user/papers/deep_learning.pdf",
    "file_name": "深度学习综述"
  }' | jq .

# 查询文档
curl -X POST http://localhost:8000/mcp/query_documents \
  -H "Content-Type: application/json" \
  -d '{
    "query": "什么是卷积神经网络？",
    "top_k": 3
  }' | jq .
```

---

## 故障排除

### 常见问题

#### 1. 文档处理失败

**问题**: OCR识别率低
**解决方案**:
- 提高原文档分辨率
- 确保文字清晰对比度高
- 检查文档是否为扫描版本

#### 2. 查询无结果

**问题**: 查询返回空结果
**解决方案**:
- 确认已有相关文档被处理
- 调整查询关键词
- 增加top_k参数值
- 检查会话ID是否正确

#### 3. 处理超时

**问题**: 大文档处理超时
**解决方案**:
- 分页处理大文档
- 增加处理超时时间
- 检查系统资源使用情况

#### 4. 内存不足

**问题**: 处理过程中内存溢出
**解决方案**:
- 减少并发处理数量
- 清理不必要的缓存
- 升级系统内存

### 调试模式

启用详细日志:
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# 处理时会输出详细日志
result = process_document(file_path="/path/to/doc.pdf")
```

### 性能监控

检查系统状态:
```bash
# 检查系统健康状态
curl http://localhost:8000/health

# 查看处理队列状态  
curl http://localhost:8000/api/status
```

---

## 版本更新日志

### v1.2.0 (2024-01-15)
- 新增结构化内容展示功能
- 优化混合检索性能
- 增加批量处理支持

### v1.1.0 (2024-01-01)  
- 添加知识图谱提取
- 支持多语言文档处理
- 改进错误处理机制

### v1.0.0 (2023-12-01)
- 初始发布版本
- 基础MCP工具支持
- 文档处理和查询功能

---

*最后更新: 2024-01-15*
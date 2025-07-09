# 学术文献OCR电子化与智能检索系统 - 命令行界面

本目录包含系统的命令行界面 (CLI) 组件，提供文档处理和智能对话的用户交互功能。

## 组件概述

CLI组件分为两个主要部分：

1. **文档处理CLI** (`document_cli.py`)  
   提供文档上传、处理、查询和导出功能。

2. **聊天对话CLI** (`chat_cli.py`)  
   提供基于文档内容的智能问答功能。

## CLI与Web界面的关系

虽然系统现在提供了Web界面（`webapp.py`），但命令行界面仍然具有以下优势：

- 支持自动化脚本和批处理
- 适合集成到其他系统工作流程中
- 提供更细粒度的控制选项
- 适合服务器环境或无图形界面场景
- 支持高级功能调试和开发

Web界面和CLI共享相同的后端逻辑和功能，只是交互方式不同。用户可以根据具体需求选择合适的交互方式。

## 文档处理CLI

文档处理CLI是系统的主要入口点，用于管理文档的整个生命周期。

### 功能特点

- 支持单个或批量文档上传与处理
- 提供文档处理进度跟踪
- 支持查询文档信息和处理状态
- 能够列出所有处理过的文档
- 支持将处理结果导出为不同格式
- 提供文档删除功能

### 使用方法

基本语法：

```bash
python document_cli.py <命令> [选项]
```

主要命令：

- `upload`: 上传并处理文档
  ```bash
  # 处理单个文档
  python document_cli.py upload --file path/to/document.pdf
  
  # 批量处理目录中的文档
  python document_cli.py upload --directory path/to/documents --recursive
  ```

- `process`: 重新处理已上传文档
  ```bash
  python document_cli.py process --id document_id --processors OCRProcessor,StructureProcessor
  ```

- `info`: 查询文档信息
  ```bash
  python document_cli.py info --id document_id
  ```

- `list`: 列出所有已处理文档
  ```bash
  # 基本列表
  python document_cli.py list
  
  # 按状态筛选
  python document_cli.py list --status completed
  
  # 按标签筛选
  python document_cli.py list --tag academic
  
  # 输出为JSON格式
  python document_cli.py list --format json
  ```

- `export`: 导出处理结果
  ```bash
  python document_cli.py export --id document_id --format markdown --output path/to/output.md
  ```

- `delete`: 删除文档
  ```bash
  python document_cli.py delete --id document_id
  ```

全局选项：

- `--config`: 指定配置文件路径
- `--verbose` 或 `-v`: 显示详细日志

### 示例

```bash
# 上传并处理PDF文档
python document_cli.py upload --file documents/paper.pdf

# 批量处理目录中的图片文件
python document_cli.py upload --directory documents/images --extensions jpg,png

# 查看特定文档的信息
python document_cli.py info --id 3fa85f64-5717-4562-b3fc-2c963f66afa6

# 导出为Markdown格式
python document_cli.py export --id 3fa85f64-5717-4562-b3fc-2c963f66afa6 --format markdown
```

## 聊天对话CLI

聊天对话CLI提供基于文档内容的交互式对话功能，支持自然语言提问和智能回答。

### 功能特点

- 提供交互式聊天界面
- 支持基于RAG技术的智能问答
- 保存和管理聊天会话历史
- 支持导出会话记录
- 提供会话列表功能

### 使用方法

基本语法：

```bash
python chat_cli.py [选项]
```

主要选项：

- `--session`: 继续已有会话
  ```bash
  python chat_cli.py --session session_id
  ```

- `--replay`: 回放会话历史
  ```bash
  python chat_cli.py --session session_id --replay
  ```

- `--list`: 列出所有会话
  ```bash
  python chat_cli.py --list
  ```

- `--export`: 导出会话记录
  ```bash
  python chat_cli.py --export session_id
  ```

其他选项：

- `--config`: 指定配置文件路径
- `--verbose` 或 `-v`: 显示详细日志

### 聊天命令

在交互式聊天模式中，支持以下命令：

- `help` 或 `?`: 显示帮助信息
- `clear` 或 `cls`: 清屏
- `save`: 保存当前会话
- `exit` 或 `quit`: 退出聊天

### 示例

```bash
# 启动新的聊天会话
python chat_cli.py

# 继续特定会话
python chat_cli.py --session 3fa85f64-5717-4562-b3fc-2c963f66afa6

# 导出会话记录
python chat_cli.py --export 3fa85f64-5717-4562-b3fc-2c963f66afa6

# 列出所有会话
python chat_cli.py --list
```

## 错误处理

两个CLI组件都实现了健壮的错误处理机制，可以处理以下情况：

- 文件不存在或无法访问
- 无效的文档格式
- 处理过程中的异常
- 无效的ID或命令
- 配置错误

遇到错误时，系统会输出清晰的错误消息，并记录详细的错误日志（使用`--verbose`选项可查看）。

## 环境要求

- Python 3.8+
- 系统依赖库（详见项目根目录的`requirements.txt`）
- 足够的磁盘空间用于存储文档和处理结果

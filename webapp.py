"""
Academic RAG Server Web Application

Provides a web interface for document upload, processing, retrieval, and intelligent chat.
"""

import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

from flask import Flask, render_template, request, redirect, url_for, jsonify, session, flash
from loguru import logger
from werkzeug.utils import secure_filename

from core.config_manager import ConfigManager
from core.pipeline import Pipeline
from models.document import Document
from rag.chat_session import ChatSession, ChatSessionManager
from rag.haystack_pipeline import RAGPipeline
from processors.pre_processor import PreProcessor
from processors.ocr_processor import OCRProcessor
from processors.structure_processor import StructureProcessor
from processors.classification_processor import ClassificationProcessor
from processors.format_converter import FormatConverter
from processors.haystack_embedding_processor import HaystackEmbeddingProcessor
import threading
import json

# Initialize Flask application
app = Flask(__name__)

# Initialize security enhancements
try:
    from utils.security_enhancements import init_security_enhancements

    security_components = init_security_enhancements(app)
except ImportError:
    # Fallback to basic security configuration
    import secrets

    app.secret_key = os.environ.get("SECRET_KEY") or secrets.token_urlsafe(32)
    logger.warning("Security enhancements not available, using basic configuration")

# Configuration
app.config["UPLOAD_FOLDER"] = os.environ.get("UPLOAD_FOLDER", "./uploads")
app.config["MAX_CONTENT_LENGTH"] = int(os.environ.get("MAX_UPLOAD_SIZE", 16)) * 1024 * 1024

# Ensure necessary directories exist
for directory in [app.config["UPLOAD_FOLDER"], "./data", "./output"]:
    Path(directory).mkdir(parents=True, exist_ok=True)

# Initialize configuration manager
config_manager = ConfigManager("./config/config.json")

# 初始化日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("webapp")

# 初始化会话管理器
session_manager = ChatSessionManager()

# 初始化处理流水线
processing_pipeline = None
rag_pipeline = None
# 使用线程安全的状态管理
from threading import Lock
import threading

document_status = {}  # 存储文档处理状态
document_status_lock = Lock()  # 保护文档状态的线程锁


def init_pipeline():
    """初始化处理流水线"""
    global processing_pipeline, rag_pipeline
    try:
        # 创建处理器
        pre_processor = PreProcessor()
        ocr_processor = OCRProcessor()
        structure_processor = StructureProcessor()
        classification_processor = ClassificationProcessor()
        format_converter = FormatConverter()
        embedding_processor = HaystackEmbeddingProcessor()

        # 设置处理器配置
        pre_processor.set_config(config_manager.get_value("processors.PreProcessor", {}))
        ocr_processor.set_config(config_manager.get_value("processors.OCRProcessor", {}))
        structure_processor.set_config(config_manager.get_value("processors.StructureProcessor", {}))
        classification_processor.set_config(config_manager.get_value("processors.ClassificationProcessor", {}))
        format_converter.set_config(config_manager.get_value("processors.FormatConverter", {}))
        embedding_processor.set_config(config_manager.get_value("processors.HaystackEmbeddingProcessor", {}))

        # 创建流水线
        processing_pipeline = Pipeline("WebPipeline")
        processing_pipeline.add_processor(pre_processor)
        processing_pipeline.add_processor(ocr_processor)
        processing_pipeline.add_processor(structure_processor)
        processing_pipeline.add_processor(classification_processor)
        processing_pipeline.add_processor(format_converter)
        processing_pipeline.add_processor(embedding_processor)

        # 初始化RAG管道
        rag_pipeline = RAGPipeline()
        rag_pipeline.init_retriever(config_manager.get_value("retriever.index_name", "academic_docs"))
        rag_pipeline.init_generator(
            config_manager.get_value("generator.model_name", "gpt-3.5-turbo"),
            config_manager.get_value("generator.params", {}),
        )

        logger.info("处理流水线和RAG系统初始化成功")
    except Exception as e:
        logger.error(f"系统初始化失败: {str(e)}")


# 初始化系统
init_pipeline()


def extract_structured_content(text: str, metadata: dict) -> dict:
    """
    从引用文本中提取结构化内容

    Args:
        text: 引用文本
        metadata: 元数据信息

    Returns:
        包含结构化内容的字典，如果没有则返回None
    """
    # 检查是否包含表格标记
    if "表格" in text or "Table" in text or "|" in text:
        # 尝试解析表格内容
        table_data = parse_table_from_text(text)
        if table_data:
            return {"type": "table", "title": metadata.get("table_caption", "数据表格"), "data": table_data}

    # 检查是否包含代码块
    if "```" in text or "def " in text or "function " in text or "class " in text:
        # 提取代码内容
        code_data = extract_code_block(text)
        if code_data:
            return {
                "type": "code",
                "title": "代码示例",
                "data": code_data["code"],
                "language": code_data.get("language", "python"),
            }

    # 检查是否包含图表引用
    if "图" in text or "Figure" in text or "Fig." in text:
        # 提取图表信息
        figure_info = extract_figure_info(text, metadata)
        if figure_info:
            return {"type": "figure", "title": figure_info.get("caption", "图表"), "data": figure_info}

    # 检查是否包含数学公式
    if "$" in text or "\\(" in text or "\\[" in text or "equation" in text:
        # 提取公式
        equation = extract_equation(text)
        if equation:
            return {"type": "equation", "title": "数学公式", "data": equation}

    return None


def parse_table_from_text(text: str) -> dict:
    """
    从文本中解析表格数据
    """
    import re

    # 查找Markdown表格
    lines = text.strip().split("\n")
    table_lines = []
    in_table = False

    for line in lines:
        if "|" in line:
            table_lines.append(line)
            in_table = True
        elif in_table and line.strip() == "":
            break

    if len(table_lines) >= 3:  # 至少需要表头、分隔符和一行数据
        try:
            # 解析表头
            headers = [cell.strip() for cell in table_lines[0].split("|") if cell.strip()]

            # 解析数据行
            rows = []
            for line in table_lines[2:]:  # 跳过分隔符行
                if "|" in line:
                    row = [cell.strip() for cell in line.split("|") if cell.strip()]
                    if len(row) == len(headers):
                        rows.append(row)

            if headers and rows:
                return {"headers": headers, "rows": rows}
        except:
            pass

    return None


def extract_code_block(text: str) -> dict:
    """
    从文本中提取代码块
    """
    import re

    # 查找带语言标记的代码块
    code_pattern = r"```(\w+)?\n([\s\S]*?)```"
    match = re.search(code_pattern, text)
    if match:
        language = match.group(1) or "text"
        code = match.group(2).strip()
        return {"code": code, "language": language}

    # 查找缩进的代码块
    lines = text.split("\n")
    code_lines = []
    for line in lines:
        if line.startswith("    ") or line.startswith("\t"):
            code_lines.append(line[4:] if line.startswith("    ") else line[1:])

    if code_lines:
        return {"code": "\n".join(code_lines), "language": "text"}

    return None


def extract_figure_info(text: str, metadata: dict) -> dict:
    """
    提取图表信息
    """
    import re

    # 尝试提取图表标题
    caption_patterns = [r"图\s*(\d+)[：:]\s*(.+)", r"Figure\s*(\d+)[：:]\s*(.+)", r"Fig\.\s*(\d+)[：:]\s*(.+)"]

    for pattern in caption_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return {
                "number": match.group(1),
                "caption": match.group(2).strip(),
                "url": metadata.get("figure_url"),  # 如果有的话
            }

    # 如果没有匹配，但文本中提到了图，返回基本信息
    if "图" in text or "figure" in text.lower():
        return {"caption": text[:100] + "..." if len(text) > 100 else text}

    return None


def extract_equation(text: str) -> str:
    """
    从文本中提取数学公式
    """
    import re

    # LaTeX内联公式
    inline_pattern = r"\$([^$]+)\$"
    match = re.search(inline_pattern, text)
    if match:
        return match.group(1)

    # LaTeX显示公式
    display_patterns = [r"\\\[([\s\S]*?)\\\]", r"\$\$([\s\S]*?)\$\$", r"\\begin{equation}([\s\S]*?)\\end{equation}"]

    for pattern in display_patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()

    # 查找可能的公式文本
    if "equation" in text.lower() or "=" in text:
        # 提取包含等号的行
        lines = text.split("\n")
        for line in lines:
            if "=" in line and not line.strip().startswith("#"):
                return line.strip()

    return None


# 允许的文件扩展名
ALLOWED_EXTENSIONS = {"pdf", "png", "jpg", "jpeg", "tif", "tiff"}


def allowed_file(filename):
    """检查文件扩展名是否在允许列表中"""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def process_document_async(document, doc_id):
    """异步处理文档"""
    global document_status, processing_pipeline
    try:
        document_status[doc_id] = "processing"
        logger.info(f"开始处理文档: {doc_id}")

        if processing_pipeline:
            result = processing_pipeline.process_document(document)
            if result.is_successful():
                document_status[doc_id] = "completed"
                logger.info(f"文档处理完成: {doc_id}")
            else:
                document_status[doc_id] = "failed"
                logger.error(f"文档处理失败: {doc_id} - {result.get_message()}")
        else:
            document_status[doc_id] = "failed"
            logger.error(f"处理流水线未初始化: {doc_id}")
    except Exception as e:
        document_status[doc_id] = "failed"
        logger.error(f"处理文档异常: {doc_id} - {str(e)}")


def format_file_size(size_bytes):
    """格式化文件大小"""
    if size_bytes == 0:
        return "0B"
    size_names = ["B", "KB", "MB", "GB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    return f"{size_bytes:.1f}{size_names[i]}"


@app.route("/")
def index():
    """首页路由"""
    return render_template("index.html")


@app.route("/documents")
def document_list():
    """文档列表页面"""
    # 这里应该从数据库或文件系统获取已处理的文档列表
    # 简化版本，仅读取上传目录中的文件
    documents = []
    try:
        for filename in os.listdir(app.config["UPLOAD_FOLDER"]):
            if allowed_file(filename):
                doc_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
                doc_size = os.path.getsize(doc_path)
                doc_time = os.path.getmtime(doc_path)
                # 尝试获取文档处理状态
                status = "已上传"
                for doc_id, doc_status in document_status.items():
                    if filename in doc_id or doc_id in filename:
                        if doc_status == "processing":
                            status = "处理中"
                        elif doc_status == "completed":
                            status = "已完成"
                        elif doc_status == "failed":
                            status = "处理失败"
                        break

                documents.append(
                    {
                        "id": filename,
                        "name": filename,
                        "size": format_file_size(doc_size),
                        "date": datetime.fromtimestamp(doc_time).strftime("%Y-%m-%d %H:%M:%S"),
                        "status": status,
                    }
                )
    except Exception as e:
        logger.error(f"读取文档列表失败: {str(e)}")
        flash("读取文档列表失败", "error")

    return render_template("documents.html", documents=documents)


@app.route("/upload", methods=["GET", "POST"])
def upload_document():
    """文档上传页面和处理"""
    if request.method == "POST":
        # 检查是否有文件
        if "file" not in request.files:
            flash("没有选择文件", "error")
            return redirect(request.url)

        file = request.files["file"]

        # 如果用户未选择文件，浏览器也会提交一个空文件
        if file.filename == "":
            flash("没有选择文件", "error")
            return redirect(request.url)

        if file and allowed_file(file.filename):
            # 安全处理文件名并保存
            filename = secure_filename(file.filename)
            # 添加时间戳避免重名
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            filename = f"{timestamp}_{filename}"

            file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(file_path)

            # 创建文档对象并添加到处理队列
            try:
                document = Document(file_path)
                doc_id = document.document_id

                # 启动异步处理任务
                thread = threading.Thread(target=process_document_async, args=(document, doc_id))
                thread.daemon = True
                thread.start()

                flash(f'文件"{filename}"上传成功，正在处理中...', "success")
            except Exception as e:
                logger.error(f"创建文档对象失败: {str(e)}")
                flash("文件上传失败", "error")

            return redirect(url_for("document_list"))

    return render_template("upload.html")


@app.route("/chat")
def chat_interface():
    """聊天界面"""
    # 检查是否已有会话ID，没有则创建新会话
    if "session_id" not in session:
        session_id = str(uuid.uuid4())
        session["session_id"] = session_id
        # 创建新会话
        chat_session = session_manager.create_session(
            session_id=session_id, metadata={"source": "web_interface"}, max_history_length=10
        )
    else:
        session_id = session["session_id"]
        # 获取现有会话，如果不存在则创建新会话
        chat_session = session_manager.get_session(session_id)
        if not chat_session:
            chat_session = session_manager.create_session(
                session_id=session_id, metadata={"source": "web_interface"}, max_history_length=10
            )

    # 获取会话历史
    messages = chat_session.messages if chat_session else []

    return render_template("chat.html", messages=messages)


@app.route("/api/chat", methods=["POST"])
def chat_query():
    """处理聊天请求API"""
    data = request.json
    query = data.get("query", "").strip()

    if not query:
        return jsonify({"error": "查询内容不能为空"}), 400

    # 获取会话ID，如果没有则创建新会话
    session_id = session.get("session_id")
    if not session_id:
        session_id = str(uuid.uuid4())
        session["session_id"] = session_id
        chat_session = session_manager.create_session(
            session_id=session_id, metadata={"source": "web_interface"}, max_history_length=10
        )
    else:
        chat_session = session_manager.get_session(session_id)
        if not chat_session:
            chat_session = session_manager.create_session(
                session_id=session_id, metadata={"source": "web_interface"}, max_history_length=10
            )

    try:
        # 记录用户消息
        chat_session.add_message("user", query)

        # 处理查询
        response = chat_session.query(query)

        # 提取回复内容
        answer = response.get("answer", "抱歉，我无法回答这个问题")
        citations = response.get("citations", [])

        # 转换引用格式为前端可用格式
        formatted_citations = []
        for citation in citations:
            formatted_citation = {
                "document_id": citation.get("document_id", ""),
                "text": citation.get("text", ""),
                "metadata": citation.get("metadata", {}),
            }

            # 提取结构化内容（表格、代码、图表等）
            structured_content = extract_structured_content(citation.get("text", ""), citation.get("metadata", {}))
            if structured_content:
                formatted_citation["structured_content"] = structured_content

            formatted_citations.append(formatted_citation)

        return jsonify({"answer": answer, "citations": formatted_citations})

    except Exception as e:
        logger.error(f"处理查询失败: {str(e)}")
        return jsonify({"error": f"处理查询失败: {str(e)}"}), 500


@app.route("/about")
def about():
    """关于页面"""
    return render_template("about.html")


@app.route("/api/document/status/<doc_id>")
def get_document_status(doc_id):
    """获取文档处理状态API"""
    status = document_status.get(doc_id, "unknown")
    return jsonify({"status": status})


@app.route("/api/document/delete/<doc_id>", methods=["DELETE"])
def delete_document(doc_id):
    """删除文档API"""
    try:
        # 删除文件
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], doc_id)
        if os.path.exists(file_path):
            os.remove(file_path)

        # 清除状态
        if doc_id in document_status:
            del document_status[doc_id]

        return jsonify({"success": True})
    except Exception as e:
        logger.error(f"删除文档失败: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/document/info/<doc_id>")
def get_document_info(doc_id):
    """获取文档详细信息API"""
    try:
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], doc_id)
        if not os.path.exists(file_path):
            return jsonify({"error": "文档不存在"}), 404

        stat = os.stat(file_path)
        info = {
            "id": doc_id,
            "name": doc_id,
            "size": format_file_size(stat.st_size),
            "date": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
            "status": document_status.get(doc_id, "unknown"),
        }

        return jsonify(info)
    except Exception as e:
        logger.error(f"获取文档信息失败: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.errorhandler(404)
def not_found_error(error):
    """404错误处理"""
    return render_template("404.html"), 404


@app.errorhandler(500)
def internal_error(error):
    """500错误处理"""
    logger.error(f"内部服务器错误: {str(error)}")
    return render_template("500.html"), 500


if __name__ == "__main__":
    # 在开发环境中使用调试模式
    debug_mode = os.environ.get("FLASK_ENV") == "development"
    app.run(host="0.0.0.0", port=5000, debug=debug_mode)

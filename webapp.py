"""
学术文献RAG服务器 Web应用

提供Web界面，用于文献上传、处理、检索和智能对话。
"""

import os
import logging
import uuid
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, jsonify, session, flash
from werkzeug.utils import secure_filename

from core.config_manager import ConfigManager
from core.pipeline import Pipeline
from models.document import Document
from rag.chat_session import ChatSession, ChatSessionManager
from rag.haystack_pipeline import RAGPipeline

# 初始化Flask应用
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev_key_for_development_only')
app.config['UPLOAD_FOLDER'] = os.environ.get('UPLOAD_FOLDER', './uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 限制上传文件大小为16MB

# 确保上传目录存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# 初始化配置管理器
config_manager = ConfigManager('./config/config.json')

# 初始化日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('webapp')

# 初始化会话管理器
session_manager = ChatSessionManager()

# 初始化RAG管道
rag_pipeline = RAGPipeline()
try:
    rag_pipeline.init_retriever(config_manager.get_value("retriever.index_name", "academic_docs"))
    rag_pipeline.init_generator(
        config_manager.get_value("generator.model_name", "gpt-3.5-turbo"),
        config_manager.get_value("generator.params", {})
    )
    logger.info("RAG系统初始化成功")
except Exception as e:
    logger.error(f"RAG系统初始化失败: {str(e)}")

# 允许的文件扩展名
ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg', 'tif', 'tiff'}

def allowed_file(filename):
    """检查文件扩展名是否在允许列表中"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """首页路由"""
    return render_template('index.html')

@app.route('/documents')
def document_list():
    """文档列表页面"""
    # 这里应该从数据库或文件系统获取已处理的文档列表
    # 简化版本，仅读取上传目录中的文件
    documents = []
    try:
        for filename in os.listdir(app.config['UPLOAD_FOLDER']):
            if allowed_file(filename):
                doc_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                doc_size = os.path.getsize(doc_path)
                doc_time = os.path.getmtime(doc_path)
                documents.append({
                    'id': filename,
                    'name': filename,
                    'size': doc_size,
                    'date': datetime.fromtimestamp(doc_time).strftime('%Y-%m-%d %H:%M:%S'),
                    'status': '已上传'  # 实际系统中应该从处理状态中读取
                })
    except Exception as e:
        logger.error(f"读取文档列表失败: {str(e)}")
        flash('读取文档列表失败', 'error')
    
    return render_template('documents.html', documents=documents)

@app.route('/upload', methods=['GET', 'POST'])
def upload_document():
    """文档上传页面和处理"""
    if request.method == 'POST':
        # 检查是否有文件
        if 'file' not in request.files:
            flash('没有选择文件', 'error')
            return redirect(request.url)
        
        file = request.files['file']
        
        # 如果用户未选择文件，浏览器也会提交一个空文件
        if file.filename == '':
            flash('没有选择文件', 'error')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            # 安全处理文件名并保存
            filename = secure_filename(file.filename)
            # 添加时间戳避免重名
            timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
            filename = f"{timestamp}_{filename}"
            
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            # 创建文档对象并添加到处理队列
            # 实际项目中这里应该启动异步处理任务
            try:
                document = Document(file_path)
                # TODO: 将文档添加到处理队列，实际处理在后台进行
                flash(f'文件"{filename}"上传成功，正在排队处理', 'success')
            except Exception as e:
                logger.error(f"处理文档失败: {str(e)}")
                flash('文件上传成功，但处理失败', 'error')
            
            return redirect(url_for('document_list'))
    
    return render_template('upload.html')

@app.route('/chat')
def chat_interface():
    """聊天界面"""
    # 检查是否已有会话ID，没有则创建新会话
    if 'session_id' not in session:
        session_id = str(uuid.uuid4())
        session['session_id'] = session_id
        # 创建新会话
        chat_session = session_manager.create_session(
            session_id=session_id,
            metadata={"source": "web_interface"},
            max_history_length=10
        )
    else:
        session_id = session['session_id']
        # 获取现有会话，如果不存在则创建新会话
        chat_session = session_manager.get_session(session_id)
        if not chat_session:
            chat_session = session_manager.create_session(
                session_id=session_id,
                metadata={"source": "web_interface"},
                max_history_length=10
            )
    
    # 获取会话历史
    messages = chat_session.messages if chat_session else []
    
    return render_template('chat.html', messages=messages)

@app.route('/api/chat', methods=['POST'])
def chat_query():
    """处理聊天请求API"""
    data = request.json
    query = data.get('query', '').strip()
    
    if not query:
        return jsonify({"error": "查询内容不能为空"}), 400
    
    # 获取会话ID，如果没有则创建新会话
    session_id = session.get('session_id')
    if not session_id:
        session_id = str(uuid.uuid4())
        session['session_id'] = session_id
        chat_session = session_manager.create_session(
            session_id=session_id,
            metadata={"source": "web_interface"},
            max_history_length=10
        )
    else:
        chat_session = session_manager.get_session(session_id)
        if not chat_session:
            chat_session = session_manager.create_session(
                session_id=session_id,
                metadata={"source": "web_interface"},
                max_history_length=10
            )
    
    try:
        # 记录用户消息
        chat_session.add_message("user", query)
        
        # 处理查询
        response = chat_session.query(query)
        
        # 提取回复内容
        answer = response.get('answer', '抱歉，我无法回答这个问题')
        citations = response.get('citations', [])
        
        # 转换引用格式为前端可用格式
        formatted_citations = []
        for citation in citations:
            formatted_citations.append({
                'document_id': citation.get('document_id', ''),
                'text': citation.get('text', ''),
                'metadata': citation.get('metadata', {})
            })
        
        return jsonify({
            "answer": answer,
            "citations": formatted_citations
        })
    
    except Exception as e:
        logger.error(f"处理查询失败: {str(e)}")
        return jsonify({"error": f"处理查询失败: {str(e)}"}), 500

@app.route('/about')
def about():
    """关于页面"""
    return render_template('about.html')

if __name__ == '__main__':
    # 在开发环境中使用调试模式
    debug_mode = os.environ.get('FLASK_ENV') == 'development'
    app.run(host='0.0.0.0', port=5000, debug=debug_mode)

/**
 * 学术文献RAG系统 - 前端脚本
 */

// 页面加载完成后执行
document.addEventListener('DOMContentLoaded', function() {
    // 激活当前导航项
    activateCurrentNavItem();
    
    // 初始化工具提示
    initializeTooltips();
    
    // 添加表单验证
    validateForms();
    
    // 初始化特定页面功能
    const currentPath = window.location.pathname;
    if (currentPath === '/' || currentPath === '/index') {
        initializeHomePage();
    } else if (currentPath === '/documents') {
        initializeDocumentsPage();
    } else if (currentPath === '/upload') {
        initializeUploadPage();
    } else if (currentPath === '/chat') {
        initializeChatPage();
    }
});

/**
 * 激活当前导航项
 */
function activateCurrentNavItem() {
    const currentPath = window.location.pathname;
    const navLinks = document.querySelectorAll('.navbar-nav .nav-link');
    
    navLinks.forEach(link => {
        const href = link.getAttribute('href');
        if (href === currentPath || 
            (currentPath === '/' && href === '/') || 
            (currentPath.startsWith(href) && href !== '/')) {
            link.classList.add('active');
            link.setAttribute('aria-current', 'page');
        }
    });
}

/**
 * 初始化Bootstrap工具提示
 */
function initializeTooltips() {
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl)
    });
}

/**
 * 表单验证
 */
function validateForms() {
    const forms = document.querySelectorAll('.needs-validation');
    
    Array.from(forms).forEach(form => {
        form.addEventListener('submit', event => {
            if (!form.checkValidity()) {
                event.preventDefault();
                event.stopPropagation();
            }
            
            form.classList.add('was-validated');
        }, false);
    });
}

/**
 * 首页功能初始化
 */
function initializeHomePage() {
    // 添加首页特定功能
    console.log('Home page initialized');
}

/**
 * 文档列表页功能初始化
 */
function initializeDocumentsPage() {
    // 搜索功能
    const searchInput = document.getElementById('searchInput');
    if (searchInput) {
        searchInput.addEventListener('keyup', function() {
            const searchValue = this.value.toLowerCase();
            const tableRows = document.querySelectorAll('tbody tr');
            
            tableRows.forEach(row => {
                const nameCell = row.querySelector('td:first-child');
                if (!nameCell) return;
                
                const docName = nameCell.textContent.toLowerCase();
                if (docName.includes(searchValue)) {
                    row.style.display = '';
                } else {
                    row.style.display = 'none';
                }
            });
        });
    }
    
    // 文档列表中的删除功能
    const deleteButtons = document.querySelectorAll('.delete-document');
    deleteButtons.forEach(button => {
        button.addEventListener('click', function(e) {
            e.preventDefault();
            const docId = this.getAttribute('data-doc-id');
            if (confirm('确定要删除此文档吗？此操作不可撤销。')) {
                deleteDocument(docId);
            }
        });
    });
    
    // 文档详情查看功能
    const infoButtons = document.querySelectorAll('.info-document');
    infoButtons.forEach(button => {
        button.addEventListener('click', function(e) {
            e.preventDefault();
            const docId = this.getAttribute('data-doc-id');
            showDocumentInfo(docId);
        });
    });
    
    // 定期更新文档状态
    setInterval(updateDocumentStatuses, 5000);
    
    // 页面加载时立即更新一次状态
    updateDocumentStatuses();
}

/**
 * 上传页面功能初始化
 */
function initializeUploadPage() {
    const fileInput = document.getElementById('fileInput');
    const previewContainer = document.getElementById('previewContainer');
    const filePreview = document.getElementById('filePreview');
    const fileDetails = document.getElementById('fileDetails');
    
    if (fileInput && previewContainer && filePreview && fileDetails) {
        fileInput.addEventListener('change', function(event) {
            const file = event.target.files[0];
            if (!file) {
                previewContainer.style.display = 'none';
                return;
            }
            
            previewContainer.style.display = 'block';
            
            // 显示文件信息
            const fileSize = file.size;
            const fileSizeFormatted = formatFileSize(fileSize);
            fileDetails.textContent = `${file.name} (${fileSizeFormatted})`;
            
            // 检查文件大小限制 (16MB)
            if (fileSize > 16 * 1024 * 1024) {
                fileDetails.innerHTML += '<br><span class="text-danger">⚠️ 文件大小超过16MB限制</span>';
            }
            
            // 预览区域
            filePreview.innerHTML = '';
            
            if (file.type.startsWith('image/')) {
                const img = document.createElement('img');
                img.classList.add('img-fluid', 'mb-2');
                img.style.maxHeight = '200px';
                img.file = file;
                
                const reader = new FileReader();
                reader.onload = (function(aImg) {
                    return function(e) {
                        aImg.src = e.target.result;
                    };
                })(img);
                reader.readAsDataURL(file);
                
                filePreview.appendChild(img);
            } else if (file.type === 'application/pdf') {
                const icon = document.createElement('i');
                icon.classList.add('bi', 'bi-file-earmark-pdf', 'display-1', 'text-danger');
                filePreview.appendChild(icon);
            } else {
                const icon = document.createElement('i');
                icon.classList.add('bi', 'bi-file-earmark', 'display-1', 'text-secondary');
                filePreview.appendChild(icon);
            }
        });
    }
    
    // 文件拖放功能
    const uploadArea = document.querySelector('.file-upload-area');
    if (uploadArea) {
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            uploadArea.addEventListener(eventName, preventDefaults, false);
        });
        
        function preventDefaults(e) {
            e.preventDefault();
            e.stopPropagation();
        }
        
        ['dragenter', 'dragover'].forEach(eventName => {
            uploadArea.addEventListener(eventName, highlight, false);
        });
        
        ['dragleave', 'drop'].forEach(eventName => {
            uploadArea.addEventListener(eventName, unhighlight, false);
        });
        
        function highlight() {
            uploadArea.classList.add('highlight');
        }
        
        function unhighlight() {
            uploadArea.classList.remove('highlight');
        }
        
        uploadArea.addEventListener('drop', handleDrop, false);
        
        function handleDrop(e) {
            const dt = e.dataTransfer;
            const files = dt.files;
            
            if (fileInput && files.length > 0) {
                fileInput.files = files;
                // 触发change事件以更新预览
                const event = new Event('change', { bubbles: true });
                fileInput.dispatchEvent(event);
            }
        }
    }
    
    // 上传进度监控
    const uploadForm = document.querySelector('form[enctype="multipart/form-data"]');
    if (uploadForm) {
        uploadForm.addEventListener('submit', function(e) {
            const submitButton = this.querySelector('button[type="submit"]');
            if (submitButton) {
                submitButton.disabled = true;
                submitButton.innerHTML = '<i class="fas fa-spinner fa-spin"></i> 上传中...';
            }
        });
    }
}

/**
 * 聊天页面功能初始化
 */
function initializeChatPage() {
    const chatForm = document.getElementById('chatForm');
    const userInput = document.getElementById('userInput');
    const chatMessages = document.getElementById('chatMessages');
    const typingIndicator = document.getElementById('typingIndicator');
    const clearChatButton = document.getElementById('clearChat');
    
    if (chatForm && userInput && chatMessages && typingIndicator) {
        // 页面加载完成后滚动到最新消息
        chatMessages.scrollTop = chatMessages.scrollHeight;
        
        // 提交表单处理
        chatForm.addEventListener('submit', function(event) {
            event.preventDefault();
            
            const userMessage = userInput.value.trim();
            if (!userMessage) return;
            
            // 添加用户消息到聊天界面
            addMessageToChat('user', userMessage);
            
            // 清空输入框
            userInput.value = '';
            
            // 显示正在输入指示器
            typingIndicator.style.display = 'flex';
            
            // 发送API请求
            fetch('/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ query: userMessage })
            })
            .then(response => response.json())
            .then(data => {
                // 隐藏输入指示器
                typingIndicator.style.display = 'none';
                
                if (data.error) {
                    // 处理错误
                    addErrorMessage(data.error);
                } else {
                    // 添加助手回复
                    addMessageToChat('assistant', data.answer, data.citations);
                }
            })
            .catch(error => {
                // 隐藏输入指示器
                typingIndicator.style.display = 'none';
                // 处理错误
                addErrorMessage('请求处理失败，请稍后重试');
                console.error('Error:', error);
            });
        });
        
        // 清空聊天记录
        if (clearChatButton) {
            clearChatButton.addEventListener('click', function() {
                if (confirm('确定要清空当前对话吗？此操作不可撤销。')) {
                    // 只保留欢迎消息
                    chatMessages.innerHTML = `
                        <div class="message-bubble assistant-message">
                            <div>您好！我是学术文献RAG助手，我可以根据已处理的学术文献回答您的问题。请问您想了解什么？</div>
                            <div class="message-time">现在</div>
                        </div>
                        <div class="typing-indicator assistant-message" id="typingIndicator">
                            <span></span>
                            <span></span>
                            <span></span>
                        </div>
                    `;
                    
                    // 更新引用指示器变量
                    typingIndicator = document.getElementById('typingIndicator');
                }
            });
        }
    }
    
    // 添加消息到聊天界面
    function addMessageToChat(role, content, citations = []) {
        const messageElement = document.createElement('div');
        messageElement.className = `message-bubble ${role === 'user' ? 'user-message' : 'assistant-message'}`;
        
        // 消息内容
        const contentElement = document.createElement('div');
        contentElement.textContent = content;
        messageElement.appendChild(contentElement);
        
        // 时间戳
        const timeElement = document.createElement('div');
        timeElement.className = 'message-time';
        timeElement.textContent = '刚刚';
        messageElement.appendChild(timeElement);
        
        // 添加引用内容
        if (role === 'assistant' && citations && citations.length > 0) {
            citations.forEach(citation => {
                const citationElement = document.createElement('div');
                citationElement.className = 'citation';
                
                const titleElement = document.createElement('div');
                titleElement.className = 'citation-title';
                titleElement.textContent = `引用来源：${citation.metadata?.title || '未知文档'}`;
                citationElement.appendChild(titleElement);
                
                const textElement = document.createElement('div');
                textElement.textContent = citation.text;
                citationElement.appendChild(textElement);
                
                // Add structured content if present
                if (citation.structured_content) {
                    const structuredElement = createStructuredContent(citation.structured_content);
                    if (structuredElement) {
                        citationElement.appendChild(structuredElement);
                    }
                }
                
                messageElement.appendChild(citationElement);
            });
        }
        
        // 插入到聊天记录前面
        chatMessages.insertBefore(messageElement, typingIndicator);
        
        // 滚动到底部
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
    
    // 添加错误消息
    function addErrorMessage(errorText) {
        const messageElement = document.createElement('div');
        messageElement.className = 'alert alert-danger mx-3 my-2';
        messageElement.textContent = `错误: ${errorText}`;
        
        // 插入到聊天记录前面
        chatMessages.insertBefore(messageElement, typingIndicator);
        
        // 滚动到底部
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
}

/**
 * 删除文档
 */
function deleteDocument(docId) {
    fetch(`/api/document/delete/${docId}`, {
        method: 'DELETE',
        headers: {
            'Content-Type': 'application/json',
        }
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            // 从页面移除该行
            const row = document.querySelector(`tr[data-doc-id="${docId}"]`);
            if (row) {
                row.remove();
            }
            showAlert('文档删除成功', 'success');
        } else {
            showAlert('删除失败: ' + (data.error || '未知错误'), 'danger');
        }
    })
    .catch(error => {
        console.error('Error:', error);
        showAlert('删除失败: 网络错误', 'danger');
    });
}

/**
 * 显示文档详情
 */
function showDocumentInfo(docId) {
    fetch(`/api/document/info/${docId}`)
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            showAlert('获取文档信息失败: ' + data.error, 'danger');
        } else {
            // 创建模态框显示文档信息
            const modalHtml = `
                <div class="modal fade" id="docInfoModal" tabindex="-1">
                    <div class="modal-dialog">
                        <div class="modal-content">
                            <div class="modal-header">
                                <h5 class="modal-title">文档信息</h5>
                                <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                            </div>
                            <div class="modal-body">
                                <p><strong>文档ID:</strong> ${data.id}</p>
                                <p><strong>文件名:</strong> ${data.name}</p>
                                <p><strong>文件大小:</strong> ${data.size}</p>
                                <p><strong>上传时间:</strong> ${data.date}</p>
                                <p><strong>处理状态:</strong> <span class="badge bg-${getStatusBadgeClass(data.status)}">${data.status}</span></p>
                            </div>
                            <div class="modal-footer">
                                <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">关闭</button>
                            </div>
                        </div>
                    </div>
                </div>
            `;
            
            // 移除现有模态框
            const existingModal = document.getElementById('docInfoModal');
            if (existingModal) {
                existingModal.remove();
            }
            
            // 添加新模态框
            document.body.insertAdjacentHTML('beforeend', modalHtml);
            
            // 显示模态框
            const modal = new bootstrap.Modal(document.getElementById('docInfoModal'));
            modal.show();
        }
    })
    .catch(error => {
        console.error('Error:', error);
        showAlert('获取文档信息失败: 网络错误', 'danger');
    });
}

/**
 * 更新文档状态
 */
function updateDocumentStatuses() {
    const statusCells = document.querySelectorAll('.document-status');
    statusCells.forEach(cell => {
        const docId = cell.getAttribute('data-doc-id');
        if (docId) {
            fetch(`/api/document/status/${docId}`)
            .then(response => response.json())
            .then(data => {
                const statusSpan = cell.querySelector('.status-badge');
                if (statusSpan && data.status) {
                    statusSpan.textContent = getStatusText(data.status);
                    statusSpan.className = `badge bg-${getStatusBadgeClass(data.status)} status-badge`;
                }
            })
            .catch(error => {
                console.error('Error updating status:', error);
            });
        }
    });
}

/**
 * 获取状态文本
 */
function getStatusText(status) {
    const statusMap = {
        'processing': '处理中',
        'completed': '已完成',
        'failed': '处理失败',
        'unknown': '未知状态'
    };
    return statusMap[status] || status;
}

/**
 * 获取状态徽章样式类
 */
function getStatusBadgeClass(status) {
    const classMap = {
        'processing': 'warning',
        'completed': 'success',
        'failed': 'danger',
        'unknown': 'secondary'
    };
    return classMap[status] || 'secondary';
}

/**
 * 显示警告消息
 */
function showAlert(message, type = 'info') {
    const alertHtml = `
        <div class="alert alert-${type} alert-dismissible fade show" role="alert">
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        </div>
    `;
    
    // 查找或创建警告容器
    let alertContainer = document.querySelector('.alert-container');
    if (!alertContainer) {
        alertContainer = document.createElement('div');
        alertContainer.className = 'alert-container position-fixed top-0 end-0 p-3';
        alertContainer.style.zIndex = '9999';
        document.body.appendChild(alertContainer);
    }
    
    alertContainer.insertAdjacentHTML('beforeend', alertHtml);
    
    // 3秒后自动隐藏
    setTimeout(() => {
        const alerts = alertContainer.querySelectorAll('.alert');
        if (alerts.length > 0) {
            alerts[0].remove();
        }
    }, 3000);
}

/**
 * 格式化文件大小
 */
function formatFileSize(bytes) {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

/**
 * 创建结构化内容元素
 */
function createStructuredContent(content) {
    if (!content || !content.type) return null;
    
    const container = document.createElement('div');
    container.className = 'structured-content';
    
    // 添加头部
    const header = document.createElement('div');
    header.className = 'structured-content-header';
    
    const title = document.createElement('span');
    title.textContent = content.title || '结构化内容';
    header.appendChild(title);
    
    const typeLabel = document.createElement('span');
    typeLabel.className = 'structured-content-type';
    typeLabel.textContent = getContentTypeLabel(content.type);
    header.appendChild(typeLabel);
    
    container.appendChild(header);
    
    // 根据类型创建内容
    switch (content.type) {
        case 'table':
            const tableElement = createTableContent(content.data);
            if (tableElement) container.appendChild(tableElement);
            break;
            
        case 'code':
            const codeElement = createCodeContent(content.data, content.language);
            if (codeElement) container.appendChild(codeElement);
            break;
            
        case 'figure':
            const figureElement = createFigureContent(content.data);
            if (figureElement) container.appendChild(figureElement);
            break;
            
        case 'equation':
            const equationElement = createEquationContent(content.data);
            if (equationElement) container.appendChild(equationElement);
            break;
            
        default:
            const defaultElement = document.createElement('div');
            defaultElement.textContent = JSON.stringify(content.data, null, 2);
            container.appendChild(defaultElement);
    }
    
    return container;
}

/**
 * 获取内容类型标签
 */
function getContentTypeLabel(type) {
    const labels = {
        'table': '表格',
        'code': '代码',
        'figure': '图表',
        'equation': '公式'
    };
    return labels[type] || type;
}

/**
 * 创建表格内容
 */
function createTableContent(data) {
    if (!data || !data.headers || !data.rows) return null;
    
    const table = document.createElement('table');
    table.className = 'content-table';
    
    // 创建表头
    const thead = document.createElement('thead');
    const headerRow = document.createElement('tr');
    data.headers.forEach(header => {
        const th = document.createElement('th');
        th.textContent = header;
        headerRow.appendChild(th);
    });
    thead.appendChild(headerRow);
    table.appendChild(thead);
    
    // 创建表体
    const tbody = document.createElement('tbody');
    data.rows.forEach(row => {
        const tr = document.createElement('tr');
        row.forEach(cell => {
            const td = document.createElement('td');
            td.textContent = cell;
            tr.appendChild(td);
        });
        tbody.appendChild(tr);
    });
    table.appendChild(tbody);
    
    return table;
}

/**
 * 创建代码内容
 */
function createCodeContent(code, language = 'text') {
    if (!code) return null;
    
    const container = document.createElement('div');
    container.className = 'code-snippet';
    container.style.position = 'relative';
    
    // 添加复制按钮
    const copyBtn = document.createElement('button');
    copyBtn.className = 'copy-code-btn';
    copyBtn.textContent = '复制';
    copyBtn.onclick = function() {
        navigator.clipboard.writeText(code).then(() => {
            copyBtn.textContent = '已复制!';
            copyBtn.classList.add('copied');
            setTimeout(() => {
                copyBtn.textContent = '复制';
                copyBtn.classList.remove('copied');
            }, 2000);
        }).catch(err => {
            console.error('复制失败:', err);
        });
    };
    container.appendChild(copyBtn);
    
    // 添加代码内容
    const pre = document.createElement('pre');
    const codeElement = document.createElement('code');
    codeElement.textContent = code;
    if (language && language !== 'text') {
        codeElement.className = `language-${language}`;
    }
    pre.appendChild(codeElement);
    container.appendChild(pre);
    
    return container;
}

/**
 * 创建图表内容
 */
function createFigureContent(data) {
    if (!data) return null;
    
    const container = document.createElement('div');
    container.className = 'figure-content';
    
    if (data.url) {
        const img = document.createElement('img');
        img.src = data.url;
        img.alt = data.caption || '图表';
        container.appendChild(img);
    }
    
    if (data.caption) {
        const caption = document.createElement('div');
        caption.className = 'figure-caption';
        caption.textContent = data.caption;
        container.appendChild(caption);
    }
    
    return container;
}

/**
 * 创建公式内容
 */
function createEquationContent(equation) {
    if (!equation) return null;
    
    const container = document.createElement('div');
    container.className = 'equation-content';
    container.textContent = equation;
    
    // 如果页面加载了MathJax，尝试渲染
    if (window.MathJax) {
        setTimeout(() => {
            MathJax.typeset([container]);
        }, 100);
    }
    
    return container;
}

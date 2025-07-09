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
    
    // 文档列表中的删除确认
    const deleteButtons = document.querySelectorAll('.delete-document');
    deleteButtons.forEach(button => {
        button.addEventListener('click', function(e) {
            if (!confirm('确定要删除此文档吗？此操作不可撤销。')) {
                e.preventDefault();
            }
        });
    });
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
            const fileSize = (file.size / 1024 / 1024).toFixed(2);
            fileDetails.textContent = `${file.name} (${fileSize} MB)`;
            
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
                    
                    // 重置会话
                    fetch('/api/chat/reset', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        }
                    })
                    .catch(error => {
                        console.error('Error resetting chat:', error);
                    });
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

"""
格式转换处理器模块 - 提供OCR结果格式转换功能

该模块实现了一个处理器，用于将OCR处理结果转换为不同的格式，
包括Markdown和PDF格式，同时保持文档的结构和布局。
"""

import os
import re
import logging
import tempfile
from typing import Dict, List, Any, Optional, Tuple

import markdown
from markdown.extensions import Extension
import pdfkit

from models.document import Document
from models.process_result import ProcessResult
from processors.base_processor import BaseProcessor
from utils.text_utils import TextCleaner, FormatConverter, KeywordExtractor

# 配置日志
logger = logging.getLogger(__name__)

class FormatConverterProcessor(BaseProcessor):
    """
    格式转换处理器，用于将OCR结果转换为不同格式。
    
    该处理器将文档的OCR结果转换为Markdown和PDF格式，
    同时保持文档的结构、布局和特殊元素（如公式、引用等）。
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化FormatConverterProcessor对象。
        
        Args:
            config: 处理器配置，默认为空字典
        """
        super().__init__(
            name="FormatConverterProcessor",
            description="将OCR结果转换为Markdown和PDF格式",
            config=config or {}
        )
        self.output_dir = self.config.get("output_dir", "output/converted")
        self.create_pdf = self.config.get("create_pdf", True)
        self.create_markdown = self.config.get("create_markdown", True)
        self.pdf_options = self.config.get("pdf_options", {
            'page-size': 'A4',
            'margin-top': '20mm',
            'margin-right': '20mm',
            'margin-bottom': '20mm',
            'margin-left': '20mm',
            'encoding': 'UTF-8',
        })
        
        # 确保输出目录存在
        os.makedirs(self.output_dir, exist_ok=True)
    
    def process(self, document: Document) -> ProcessResult:
        """
        处理文档并返回处理结果。
        
        将文档的OCR结果转换为配置的输出格式，并存储到输出目录。
        
        Args:
            document: 要处理的Document对象
            
        Returns:
            表示处理结果的ProcessResult对象
        """
        try:
            # 获取OCR内容
            ocr_content = self._get_ocr_content(document)
            if not ocr_content:
                return ProcessResult.error_result("无可用的OCR内容进行转换")
            
            # 获取或提取文档结构
            doc_structure = document.get_content("structure")
            if not doc_structure:
                # 如果没有结构信息，尝试提取
                doc_structure = DocumentStructureExtractor.extract_structure(ocr_content)
            
            # 创建文档基础名称
            base_name = os.path.splitext(document.file_name)[0]
            base_path = os.path.join(self.output_dir, base_name)
            
            # 转换结果
            conversion_results = {}
            
            # 转换为Markdown
            if self.create_markdown:
                md_path = f"{base_path}.md"
                md_content = self._convert_to_markdown(ocr_content, doc_structure)
                
                with open(md_path, 'w', encoding='utf-8') as f:
                    f.write(md_content)
                
                conversion_results["markdown"] = {
                    "path": md_path,
                    "size": os.path.getsize(md_path)
                }
                
                # 存储Markdown内容到文档
                document.store_content("markdown", {
                    "text": md_content,
                    "path": md_path
                })
            
            # 转换为PDF
            if self.create_pdf:
                pdf_path = f"{base_path}.pdf"
                
                if self.create_markdown:
                    # 从Markdown创建PDF
                    self._convert_markdown_to_pdf(md_content, pdf_path)
                else:
                    # 直接从OCR内容创建PDF
                    temp_md_content = self._convert_to_markdown(ocr_content, doc_structure)
                    self._convert_markdown_to_pdf(temp_md_content, pdf_path)
                
                conversion_results["pdf"] = {
                    "path": pdf_path,
                    "size": os.path.getsize(pdf_path)
                }
                
                # 存储PDF路径到文档
                document.add_metadata("pdf_path", pdf_path)
            
            # 更新文档元数据
            document.add_metadata("converted_formats", list(conversion_results.keys()))
            
            return ProcessResult.success_result(
                "格式转换成功",
                conversion_results
            )
            
        except Exception as e:
            logger.error(f"格式转换处理失败: {str(e)}", exc_info=True)
            return ProcessResult.error_result(f"格式转换失败: {str(e)}", e)
    
    def _get_ocr_content(self, document: Document) -> str:
        """
        获取文档的OCR内容。
        
        Args:
            document: Document对象
            
        Returns:
            OCR文本内容
        """
        # 尝试获取OCR内容
        ocr_content = document.get_content("ocr")
        
        if ocr_content:
            if isinstance(ocr_content, str):
                return ocr_content
            elif isinstance(ocr_content, dict) and ocr_content.get("text"):
                return ocr_content.get("text")
        
        # 如果没有OCR内容，尝试获取其他文本内容
        for stage in ["structure", "preprocessed"]:
            content = document.get_content(stage)
            if content:
                if isinstance(content, str):
                    return content
                elif isinstance(content, dict) and content.get("text"):
                    return content.get("text")
        
        logger.warning(f"未找到可用的OCR内容: {document.document_id}")
        return ""
    
    def _convert_to_markdown(self, text: str, doc_structure: Dict[str, Any] = None) -> str:
        """
        将文本转换为Markdown格式。
        
        使用FormatConverter工具将OCR文本转换为Markdown格式。
        
        Args:
            text: OCR文本内容
            doc_structure: 文档结构信息
            
        Returns:
            Markdown格式文本
        """
        return FormatConverter.text_to_markdown(text, doc_structure)
    
    def _convert_markdown_to_pdf(self, md_content: str, output_path: str) -> None:
        """
        将Markdown转换为PDF。
        
        使用pdfkit将Markdown文本转换为PDF文件。
        
        Args:
            md_content: Markdown文本内容
            output_path: 输出PDF文件路径
        """
        try:
            # 首先将Markdown转换为HTML
            html_content = markdown.markdown(
                md_content,
                extensions=['extra', 'codehilite', 'tables', 'toc', MathExtension()]
            )
            
            # 添加样式
            html_template = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="UTF-8">
                <title>Converted Document</title>
                <style>
                    body {{
                        font-family: Arial, sans-serif;
                        font-size: 12pt;
                        line-height: 1.5;
                        margin: 20px;
                    }}
                    h1 {{
                        font-size: 24pt;
                        margin-top: 24pt;
                        margin-bottom: 8pt;
                    }}
                    h2 {{
                        font-size: 18pt;
                        margin-top: 18pt;
                        margin-bottom: 6pt;
                    }}
                    h3 {{
                        font-size: 14pt;
                        margin-top: 14pt;
                        margin-bottom: 4pt;
                    }}
                    p {{
                        margin-bottom: 10pt;
                    }}
                    table {{
                        border-collapse: collapse;
                        width: 100%;
                        margin-bottom: 16pt;
                    }}
                    th, td {{
                        border: 1px solid #ddd;
                        padding: 8px;
                    }}
                    th {{
                        background-color: #f2f2f2;
                    }}
                    img {{
                        max-width: 100%;
                    }}
                    .math {{
                        font-style: italic;
                    }}
                    .citation {{
                        font-weight: bold;
                    }}
                    .footnote {{
                        font-size: 10pt;
                        margin-top: 20pt;
                    }}
                    code {{
                        background-color: #f5f5f5;
                        padding: 2px 4px;
                        border-radius: 4px;
                    }}
                </style>
            </head>
            <body>
            {html_content}
            </body>
            </html>
            """
            
            # 使用临时文件存储HTML
            with tempfile.NamedTemporaryFile(suffix='.html', delete=False, mode='w', encoding='utf-8') as temp_html:
                temp_html_path = temp_html.name
                temp_html.write(html_template)
            
            try:
                # 使用pdfkit将HTML转换为PDF
                pdfkit.from_file(temp_html_path, output_path, options=self.pdf_options)
                logger.info(f"成功生成PDF: {output_path}")
            finally:
                # 删除临时HTML文件
                if os.path.exists(temp_html_path):
                    os.unlink(temp_html_path)
                
        except Exception as e:
            logger.error(f"Markdown转PDF失败: {str(e)}", exc_info=True)
            raise


class MathExtension(Extension):
    """
    Markdown的数学公式扩展，用于正确处理LaTeX数学公式。
    """
    def extendMarkdown(self, md):
        # 处理行内公式
        inline_pattern = r'\$([^$\n]+)\$'
        md.inlinePatterns.register(MathInlineProcessor(inline_pattern, md), 'math_inline', 175)
        
        # 处理块级公式
        block_pattern = r'\$\$(.*?)\$\$'
        md.inlinePatterns.register(MathBlockProcessor(block_pattern, md), 'math_block', 176)


class MathInlineProcessor(markdown.inlinepatterns.InlineProcessor):
    """
    处理行内数学公式的Markdown处理器。
    """
    def handleMatch(self, m, data):
        formula = m.group(1)
        span = m.span(0)
        elem = markdown.util.etree.Element('span')
        elem.set('class', 'math')
        elem.text = formula
        return elem, span[0], span[1]


class MathBlockProcessor(markdown.inlinepatterns.InlineProcessor):
    """
    处理块级数学公式的Markdown处理器。
    """
    def handleMatch(self, m, data):
        formula = m.group(1)
        span = m.span(0)
        elem = markdown.util.etree.Element('div')
        elem.set('class', 'math-block')
        elem.text = formula
        return elem, span[0], span[1]



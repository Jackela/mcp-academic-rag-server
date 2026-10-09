"""
格式转换处理器模块 - 提供OCR结果格式转换功能

该模块实现了一个处理器，用于将OCR处理结果转换为不同的格式，
包括Markdown和PDF格式，同时保持文档的结构和布局。
"""

import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor
from html.parser import HTMLParser
from typing import Any, Dict, Optional, Tuple, Union, overload
from xml.etree import ElementTree

import markdown
from markdown.blockprocessors import BlockProcessor
from markdown.extensions import Extension
from markdown.inlinepatterns import InlineProcessor
from playwright.sync_api import Route, sync_playwright

from models.document import Document
from models.process_result import ProcessResult
from processors.base_processor import BaseProcessor
from utils.text_utils import DocumentStructureExtractor, FormatConverter

# 配置日志
logger = logging.getLogger(__name__)


class FormatConverterProcessor(BaseProcessor):
    """
    格式转换处理器，用于将OCR结果转换为不同格式。

    该处理器将文档的OCR结果转换为Markdown和PDF格式，
    同时保持文档的结构、布局和特殊元素（如公式、引用等）。
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        初始化FormatConverterProcessor对象。

        Args:
            config: 处理器配置，默认为空字典
        """
        super().__init__(
            name="FormatConverterProcessor", description="将OCR结果转换为Markdown和PDF格式", config=config or {}
        )
        self.output_dir = self.config.get("output_dir", "output/converted")
        self.create_pdf = self.config.get("create_pdf", True)
        self.create_markdown = self.config.get("create_markdown", True)
        self.pdf_options = self.config.get(
            "pdf_options",
            {
                "page-size": "A4",
                "margin-top": "20mm",
                "margin-right": "20mm",
                "margin-bottom": "20mm",
                "margin-left": "20mm",
                "encoding": "UTF-8",
            },
        )

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

                with open(md_path, "w", encoding="utf-8") as f:
                    f.write(md_content)

                conversion_results["markdown"] = {"path": md_path, "size": os.path.getsize(md_path)}

                # 存储Markdown内容到文档
                document.store_content("markdown", {"text": md_content, "path": md_path})

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

                conversion_results["pdf"] = {"path": pdf_path, "size": os.path.getsize(pdf_path)}

                # 存储PDF路径到文档
                document.add_metadata("pdf_path", pdf_path)

            # 更新文档元数据
            document.add_metadata("converted_formats", list(conversion_results.keys()))

            return ProcessResult.success_result("格式转换成功", conversion_results)

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
            elif isinstance(ocr_content, dict) and isinstance(ocr_content.get("text"), str):
                text = ocr_content["text"]
                if isinstance(text, str):
                    return text

        # 如果没有OCR内容，尝试获取其他文本内容
        for stage in ["structure", "preprocessed"]:
            content = document.get_content(stage)
            if content:
                if isinstance(content, str):
                    return content
                elif isinstance(content, dict) and isinstance(content.get("text"), str):
                    text = content["text"]
                    if isinstance(text, str):
                        return text

        logger.warning(f"未找到可用的OCR内容: {document.document_id}")
        return ""

    def _convert_to_markdown(self, text: str, doc_structure: Optional[Dict[str, Any]] = None) -> str:
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

        使用正常Chromium将Markdown文本转换为PDF文件，禁用脚本与外部资源。

        Args:
            md_content: Markdown文本内容
            output_path: 输出PDF文件路径
        """
        try:
            # 首先将Markdown转换为HTML
            html_content = markdown.markdown(
                md_content, extensions=["extra", "codehilite", "tables", "toc", MathExtension()]
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

            options = self._pdf_render_options()
            validator = _PDFResourceValidator()
            validator.feed(html_template)
            # The sync browser API runs outside any caller's active asyncio loop.
            with ThreadPoolExecutor(max_workers=1) as executor:
                executor.submit(_render_pdf, html_template, output_path, options).result()
            logger.info(f"成功生成PDF: {output_path}")

        except Exception as e:
            logger.error(f"Markdown转PDF失败: {str(e)}", exc_info=True)
            raise

    def _pdf_render_options(self) -> Dict[str, Any]:
        allowed = {"page-size", "margin-top", "margin-right", "margin-bottom", "margin-left", "encoding", "orientation"}
        unknown = set(self.pdf_options) - allowed
        if unknown:
            raise ValueError(f"Unsupported PDF options: {sorted(unknown)}")
        if self.pdf_options.get("encoding", "UTF-8").upper().replace("-", "") != "UTF8":
            raise ValueError("PDF HTML supports UTF-8 encoding")
        orientation = self.pdf_options.get("orientation", "Portrait")
        if orientation not in {"Portrait", "Landscape"}:
            raise ValueError("PDF orientation must be Portrait or Landscape")
        return {
            "format": self.pdf_options.get("page-size", "A4"),
            "margin": {
                side: self.pdf_options.get(f"margin-{side}", "20mm") for side in ("top", "right", "bottom", "left")
            },
            "landscape": orientation == "Landscape",
        }


class _PDFResourceValidator(HTMLParser):
    """Reject local files and remote assets; self-contained data images remain usable."""

    def handle_starttag(self, tag: str, attrs: list[tuple[str, Optional[str]]]) -> None:
        for name, value in attrs:
            if name in {"src", "srcset", "data"} or (tag == "link" and name == "href"):
                if not value or not (tag == "img" and name == "src" and value.startswith("data:image/")):
                    raise ValueError(
                        "PDF resources must be self-contained data images; local and external resources are disabled"
                    )


def _render_pdf(html: str, output_path: str, options: Dict[str, Any]) -> None:
    blocked: list[str] = []

    def reject_request(route: Route) -> None:
        blocked.append("blocked")
        route.abort()

    with sync_playwright() as runtime:
        browser = runtime.chromium.launch()
        try:
            page = browser.new_page(java_script_enabled=False)
            page.route("**/*", reject_request)
            page.set_content(html, wait_until="networkidle", timeout=15000)
            if blocked:
                raise ValueError("PDF rendering cannot fetch local or external resources")
            page.pdf(path=output_path, **options)
        finally:
            browser.close()


class MathExtension(Extension):
    """
    Markdown的数学公式扩展，用于正确处理LaTeX数学公式。
    """

    def extendMarkdown(self, md: markdown.Markdown) -> None:
        # 处理行内公式
        inline_pattern = r"\$([^$\n]+)\$"
        md.inlinePatterns.register(MathInlineProcessor(inline_pattern, md), "math_inline", 175)

        # 处理块级公式
        md.parser.blockprocessors.register(MathBlockProcessor(md.parser), "math_block", 175)


class MathInlineProcessor(InlineProcessor):
    """
    处理行内数学公式的Markdown处理器。
    """

    @overload
    def handleMatch(self, m: re.Match[str]) -> ElementTree.Element: ...

    @overload
    def handleMatch(self, m: re.Match[str], data: str) -> Tuple[ElementTree.Element, int, int]: ...

    def handleMatch(
        self, m: re.Match[str], data: Optional[str] = None
    ) -> Union[ElementTree.Element, Tuple[ElementTree.Element, int, int]]:
        formula = m.group(1)
        span = m.span(0)
        elem = ElementTree.Element("span")
        elem.set("class", "math")
        elem.text = formula
        return elem if data is None else (elem, span[0], span[1])


class MathBlockProcessor(BlockProcessor):
    """Render a complete $$ formula block as a sibling of paragraphs."""

    def test(self, parent: ElementTree.Element, block: str) -> bool:
        return re.fullmatch(r"\$\$(.*?)\$\$", block.strip(), re.DOTALL) is not None

    def run(self, parent: ElementTree.Element, blocks: list[str]) -> None:
        match = re.fullmatch(r"\$\$(.*?)\$\$", blocks.pop(0).strip(), re.DOTALL)
        if match is None:
            raise ValueError("Math block requires closing $$")
        element = ElementTree.SubElement(parent, "div", {"class": "math-block"})
        element.text = match.group(1).strip()

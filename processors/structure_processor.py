
"""
结构识别处理器模块 - 实现文档结构元素识别功能

该模块提供了结构识别处理器，用于识别学术文档中的标题、摘要、章节、表格等结构元素。
"""

import re
import logging
from typing import Dict, Any, List, Optional, Tuple

from processors.base_processor import BaseProcessor
from models.document import Document
from models.process_result import ProcessResult
from utils.text_utils import DocumentStructureExtractor

# 配置日志
logger = logging.getLogger(__name__)

class StructureProcessor(BaseProcessor):
    """结构识别处理器，识别文档的结构元素"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化结构识别处理器
        
        Args:
            config (dict, optional): 处理器配置
        """
        name = "结构识别处理器"
        description = "识别文档的标题、摘要、章节等结构元素"
        super().__init__(name, description, config)
        
        # 默认配置
        self.default_config = {
            'recognize_title': True,       # 是否识别标题
            'recognize_abstract': True,    # 是否识别摘要
            'recognize_sections': True,    # 是否识别章节
            'recognize_tables': True,      # 是否识别表格
            'recognize_figures': True,     # 是否识别图表
            'recognize_references': True,  # 是否识别参考文献
            'language': 'auto',            # 文档语言，auto为自动检测
            # 正则表达式模式
            'patterns': {
                'title': r'^(?!(?:Abstract|Introduction|Conclusion|References|Bibliography|Acknowledgements))[A-Z][\w\s:,\-]+$',
                'abstract': r'(?i)abstract[:\.\s]+([\s\S]+?)(?=\n\s*(?:[1I]ntroduction|Keywords:|$))',
                'section_heading': r'^\s*(\d+(?:\.\d+)*)\s+([A-Z][^\n]+)$',
                'figure_caption': r'(?i)(?:figure|fig\.)\s+(\d+)[:\.\s]+([^\n]+)',
                'table_caption': r'(?i)(?:table)\s+(\d+)[:\.\s]+([^\n]+)',
                'references': r'(?i)(?:^|\n)references\s*(?:\n|$)([\s\S]+)',
            }
        }
        
        # 合并配置
        self.config = {**self.default_config, **(config or {})}
    
    def get_stage(self) -> str:
        """
        获取处理阶段名称
        
        Returns:
            str: 处理阶段名称
        """
        return "structure_recognition"
    
    def process(self, document: Document) -> ProcessResult:
        """
        处理文档结构识别
        
        Args:
            document (Document): 要处理的文档对象
            
        Returns:
            ProcessResult: 处理结果
        """
        try:
            logger.info(f"开始结构识别处理文档: {document.file_name}")
            
            # 更新文档状态
            document.update_status("structure_processing")
            
            # 获取OCR结果
            ocr_result = document.get_content("ocr")
            
            if not ocr_result or "text" not in ocr_result:
                return ProcessResult.error_result("无法获取OCR文本，结构识别失败")
            
            # 获取OCR文本
            text = ocr_result["text"]
            text_by_page = ocr_result.get("text_by_page", [])
            
            # 识别文档结构
            structure = self._recognize_structure(text, text_by_page)
            
            # 记录处理结果
            result_data = {
                "structure": structure,
                "language": self._detect_language(text),
                "document_type": self._detect_document_type(text, structure)
            }
            
            # 将处理结果保存到文档对象
            document.store_content(self.get_stage(), result_data)
            document.add_metadata("has_structure", True)
            
            # 添加标签
            if "document_type" in result_data:
                document.add_tag(result_data["document_type"])
            if "language" in result_data:
                document.add_tag(result_data["language"])
            
            # 提取关键元数据
            if structure.get("title"):
                document.add_metadata("title", structure["title"])
            if structure.get("abstract"):
                document.add_metadata("abstract", structure["abstract"])
            if structure.get("keywords"):
                document.add_metadata("keywords", structure["keywords"])
            if structure.get("authors"):
                document.add_metadata("authors", structure["authors"])
            
            logger.info(f"文档结构识别完成: {document.file_name}")
            return ProcessResult.success_result("文档结构识别完成", result_data)
            
        except Exception as e:
            logger.error(f"文档结构识别失败: {document.file_name}, 错误: {str(e)}", exc_info=True)
            return ProcessResult.error_result(f"结构识别时发生错误: {str(e)}", e)
    
    def _recognize_structure(self, text: str, text_by_page: List[str]) -> Dict[str, Any]:
        """
        识别文档结构（使用统一的结构提取器）
        
        Args:
            text (str): 完整文本
            text_by_page (List[str]): 按页面分组的文本
            
        Returns:
            Dict[str, Any]: 识别的结构元素
        """
        # 使用统一的结构提取器
        structure = DocumentStructureExtractor.extract_structure(text, text_by_page)
        
        # 根据配置过滤结果
        filtered_structure = {}
        
        if self.config['recognize_title'] and structure.get('title'):
            filtered_structure['title'] = structure['title']
        
        if self.config['recognize_abstract'] and structure.get('abstract'):
            filtered_structure['abstract'] = structure['abstract']
        
        if structure.get('keywords'):
            filtered_structure['keywords'] = structure['keywords']
        
        if structure.get('authors'):
            filtered_structure['authors'] = structure['authors']
        
        if self.config['recognize_sections'] and structure.get('sections'):
            filtered_structure['sections'] = structure['sections']
        
        if self.config['recognize_tables'] and structure.get('tables'):
            filtered_structure['tables'] = structure['tables']
        
        if self.config['recognize_figures'] and structure.get('figures'):
            filtered_structure['figures'] = structure['figures']
        
        if self.config['recognize_references'] and structure.get('references'):
            filtered_structure['references'] = structure['references']
        
        return filtered_structure
    
    
    def _detect_language(self, text: str) -> str:
        """
        检测文档语言
        
        Args:
            text (str): 文档文本
            
        Returns:
            str: 检测到的语言代码
        """
        try:
            # 这里使用一个简单的语言检测方法
            # 在实际应用中，可以使用语言检测库，如langdetect
            
            # 英文检测
            english_words = ['the', 'and', 'of', 'in', 'to', 'is', 'that', 'for', 'this', 'with']
            english_count = sum(1 for word in english_words if f" {word} " in f" {text.lower()} ")
            
            # 中文检测
            chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
            
            if chinese_chars > len(text) * 0.3:
                return "zh"
            elif english_count > 5:
                return "en"
            else:
                # 默认返回英文
                return "en"
                
        except Exception as e:
            logger.error(f"语言检测失败: {e}")
            return "en"  # 默认为英文
    
    def _detect_document_type(self, text: str, structure: Dict[str, Any]) -> str:
        """
        检测文档类型
        
        Args:
            text (str): 文档文本
            structure (Dict[str, Any]): 已识别的结构
            
        Returns:
            str: 文档类型
        """
        try:
            # 判断是否为学术论文
            has_abstract = bool(structure.get("abstract"))
            has_references = bool(structure.get("references"))
            
            # 查找关键词
            keywords = {
                "paper": ["study", "research", "experiment", "analysis", "methodology", "result", "conclusion"],
                "report": ["report", "assessment", "evaluation", "summary", "overview", "finding"],
                "thesis": ["thesis", "dissertation", "doctor", "phd", "master"],
                "patent": ["patent", "invention", "claim", "method of", "system for"]
            }
            
            # 计算各类型关键词出现的次数
            scores = {}
            for doc_type, words in keywords.items():
                scores[doc_type] = sum(1 for word in words if f" {word.lower()} " in f" {text.lower()} ")
            
            # 根据关键词和结构特征判断文档类型
            if has_abstract and has_references:
                if "thesis" in scores and scores["thesis"] > 3:
                    return "thesis"
                elif "patent" in scores and scores["patent"] > 3:
                    return "patent"
                elif "paper" in scores and scores["paper"] > 5:
                    return "academic_paper"
                else:
                    return "academic_paper"  # 默认为学术论文
            elif has_references:
                if scores["report"] > scores["paper"]:
                    return "technical_report"
                else:
                    return "article"
            else:
                return "document"  # 未识别类型
                
        except Exception as e:
            logger.error(f"文档类型检测失败: {e}")
            return "document"  # 默认为普通文档
    
    def _extract_tables(self, tables: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        增强的表格结构化处理，生成Markdown格式的表格表示
        
        Args:
            tables (List[Dict[str, Any]]): 原始表格数据
            
        Returns:
            List[Dict[str, Any]]: 增强处理后的表格数据
        """
        enhanced_tables = []
        
        for table in tables:
            enhanced_table = table.copy()
            
            # 如果表格有原始内容，尝试将其转换为结构化的Markdown表格
            if 'content' in table and table['content']:
                try:
                    structured_content = self._convert_table_to_markdown(table['content'])
                    enhanced_table['structured_content'] = structured_content
                    enhanced_table['content'] = structured_content  # 更新主要内容
                except Exception as e:
                    logger.warning(f"表格结构化处理失败: {e}")
                    # 保持原始内容
                    enhanced_table['structured_content'] = f"表格 {table.get('number', 'N/A')}: {table.get('caption', '无标题')}\n\n{table['content']}"
            else:
                # 如果没有表格内容，至少提供描述性文本
                enhanced_table['structured_content'] = f"表格 {table.get('number', 'N/A')}: {table.get('caption', '无标题')}"
                enhanced_table['content'] = enhanced_table['structured_content']
            
            # 添加元数据
            enhanced_table['type'] = 'table'
            enhanced_table['multimodal_type'] = 'tabular_data'
            
            enhanced_tables.append(enhanced_table)
        
        return enhanced_tables
    
    def _extract_figures(self, figures: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        增强的图表结构化处理，生成描述性文本表示
        
        Args:
            figures (List[Dict[str, Any]]): 原始图表数据
            
        Returns:
            List[Dict[str, Any]]: 增强处理后的图表数据
        """
        enhanced_figures = []
        
        for figure in figures:
            enhanced_figure = figure.copy()
            
            # 生成描述性文本表示
            try:
                structured_content = self._generate_figure_description(figure)
                enhanced_figure['structured_content'] = structured_content
                enhanced_figure['content'] = structured_content
            except Exception as e:
                logger.warning(f"图表结构化处理失败: {e}")
                # 提供基本描述
                enhanced_figure['structured_content'] = f"图 {figure.get('number', 'N/A')}: {figure.get('caption', '无标题')}"
                enhanced_figure['content'] = enhanced_figure['structured_content']
            
            # 添加元数据
            enhanced_figure['type'] = 'figure'
            enhanced_figure['multimodal_type'] = 'visual_content'
            
            enhanced_figures.append(enhanced_figure)
        
        return enhanced_figures
    
    def _convert_table_to_markdown(self, table_content: str) -> str:
        """
        将表格内容转换为Markdown格式
        
        Args:
            table_content (str): 原始表格内容
            
        Returns:
            str: Markdown格式的表格
        """
        lines = table_content.strip().split('\n')
        
        # 过滤空行
        lines = [line.strip() for line in lines if line.strip()]
        
        if not lines:
            return "空表格"
        
        # 尝试检测分隔符（制表符、多个空格、|等）
        separators = ['\t', '|', '  ', '   ', '    ']
        best_separator = None
        max_columns = 0
        
        for sep in separators:
            test_columns = len(lines[0].split(sep))
            if test_columns > max_columns and test_columns > 1:
                max_columns = test_columns
                best_separator = sep
        
        if not best_separator:
            # 如果无法检测分隔符，按单词分割第一行
            words = lines[0].split()
            if len(words) > 1:
                best_separator = ' '
            else:
                return f"表格内容:\n{table_content}"
        
        # 构建Markdown表格
        markdown_lines = []
        
        for i, line in enumerate(lines[:10]):  # 限制最多10行
            cells = [cell.strip() for cell in line.split(best_separator) if cell.strip()]
            
            if not cells:
                continue
            
            # 限制列数（最多8列）
            cells = cells[:8]
            
            # 构建Markdown行
            if i == 0:
                # 表头
                markdown_lines.append('| ' + ' | '.join(cells) + ' |')
                # 分隔行
                markdown_lines.append('| ' + ' | '.join(['---'] * len(cells)) + ' |')
            else:
                # 数据行，确保列数匹配
                while len(cells) < len(lines[0].split(best_separator)):
                    cells.append('')
                cells = cells[:len(lines[0].split(best_separator))]  # 确保不超过表头列数
                markdown_lines.append('| ' + ' | '.join(cells) + ' |')
        
        if len(lines) > 10:
            markdown_lines.append('| ... | ... | ... |')
            markdown_lines.append(f'（表格共 {len(lines)} 行，显示前10行）')
        
        return '\n'.join(markdown_lines)
    
    def _generate_figure_description(self, figure: Dict[str, Any]) -> str:
        """
        生成图表的描述性文本
        
        Args:
            figure (Dict[str, Any]): 图表信息
            
        Returns:
            str: 描述性文本
        """
        figure_num = figure.get('number', 'N/A')
        caption = figure.get('caption', '无标题')
        
        # 分析标题中的关键词来推断图表类型和内容
        caption_lower = caption.lower()
        
        # 推断图表类型
        figure_type = "图表"
        if any(word in caption_lower for word in ['flow', 'flowchart', 'workflow', '流程']):
            figure_type = "流程图"
        elif any(word in caption_lower for word in ['bar', 'histogram', '柱状', '条形']):
            figure_type = "柱状图"
        elif any(word in caption_lower for word in ['line', 'trend', '趋势', '折线']):
            figure_type = "折线图"
        elif any(word in caption_lower for word in ['pie', '饼图', '圆饼']):
            figure_type = "饼图"
        elif any(word in caption_lower for word in ['scatter', '散点']):
            figure_type = "散点图"
        elif any(word in caption_lower for word in ['network', '网络', 'graph']):
            figure_type = "网络图"
        elif any(word in caption_lower for word in ['architecture', '架构', 'structure', '结构']):
            figure_type = "架构图"
        
        # 构建描述性文本
        description = f"图 {figure_num}: {caption}\n"
        description += f"类型: {figure_type}\n"
        
        # 尝试从标题中提取更多信息
        if any(word in caption_lower for word in ['comparison', 'vs', 'versus', '比较', '对比']):
            description += "内容: 该图表显示了不同项目或条件之间的比较分析。\n"
        elif any(word in caption_lower for word in ['result', 'outcome', '结果', '效果']):
            description += "内容: 该图表展示了研究或实验的结果数据。\n"
        elif any(word in caption_lower for word in ['model', 'framework', '模型', '框架']):
            description += "内容: 该图表描述了理论模型或分析框架的结构。\n"
        elif any(word in caption_lower for word in ['process', 'procedure', '过程', '步骤']):
            description += "内容: 该图表说明了特定过程或操作步骤。\n"
        else:
            description += f"内容: {caption}\n"
        
        return description

"""
文本处理工具模块 - 提供文本分析、标签提取和格式转换功能

该模块提供了与学术文献处理相关的各种文本处理功能，包括文本清洗、
标签(关键词)提取、主题识别、格式转换等。
"""

import re
import string
import unicodedata
from typing import List, Dict, Any, Set, Tuple, Optional
import logging

# 配置日志
logger = logging.getLogger(__name__)

class TextCleaner:
    """文本清洗工具类，提供文本预处理功能"""
    
    @staticmethod
    def remove_extra_whitespace(text: str) -> str:
        """
        移除文本中的多余空白字符
        
        Args:
            text (str): 原始文本
            
        Returns:
            str: 处理后的文本
        """
        # 替换多个空白字符为单个空格
        text = re.sub(r'\s+', ' ', text)
        # 移除行首和行尾的空白字符
        return text.strip()
    
    @staticmethod
    def normalize_unicode(text: str) -> str:
        """
        Unicode标准化处理
        
        Args:
            text (str): 原始文本
            
        Returns:
            str: 标准化后的文本
        """
        return unicodedata.normalize('NFKC', text)
    
    @staticmethod
    def remove_special_chars(text: str, keep_chars: str = '') -> str:
        """
        移除特殊字符
        
        Args:
            text (str): 原始文本
            keep_chars (str): 需要保留的特殊字符
            
        Returns:
            str: 处理后的文本
        """
        # 定义要保留的字符
        pattern = r'[^\w\s' + re.escape(keep_chars) + ']'
        # 替换特殊字符为空格
        return re.sub(pattern, ' ', text)
    
    @staticmethod
    def clean_academic_text(text: str) -> str:
        """
        清洗学术文本，保留必要的学术符号
        
        Args:
            text (str): 原始学术文本
            
        Returns:
            str: 清洗后的文本
        """
        # 学术文本中需要保留的特殊字符
        academic_chars = '.,;:()[]{}"-+*/=%$@#&'
        
        # 标准化处理
        text = TextCleaner.normalize_unicode(text)
        
        # 保留学术符号的情况下移除其他特殊字符
        text = TextCleaner.remove_special_chars(text, academic_chars)
        
        # 移除多余空白字符
        text = TextCleaner.remove_extra_whitespace(text)
        
        return text
    
    @staticmethod
    def extract_paragraphs(text: str) -> List[str]:
        """
        从文本中提取段落
        
        Args:
            text (str): 原始文本
            
        Returns:
            List[str]: 段落列表
        """
        # 按照连续的多个换行符分割文本
        paragraphs = re.split(r'\n\s*\n', text)
        # 清理每个段落
        paragraphs = [TextCleaner.remove_extra_whitespace(p) for p in paragraphs if p.strip()]
        return paragraphs

    @staticmethod
    def clean_for_classification(text: str) -> str:
        """
        为文档分类准备文本
        
        Args:
            text (str): 原始文本
            
        Returns:
            str: 处理后的文本
        """
        text = TextCleaner.normalize_unicode(text)
        text = TextCleaner.remove_extra_whitespace(text)
        # 保留更多的标点符号，以保持文本的语义完整性
        text = re.sub(r'[^\w\s.,;:()[\]{}"\'-]', ' ', text)
        return text


class KeywordExtractor:
    """关键词提取工具类，提供从文本中提取关键词和主题的功能"""
    
    @staticmethod
    def extract_ngrams(text: str, n: int = 2, min_freq: int = 2) -> List[Tuple[str, int]]:
        """
        从文本中提取N元词组
        
        Args:
            text (str): 原始文本
            n (int): N元词组的N值
            min_freq (int): 最小出现频率
            
        Returns:
            List[Tuple[str, int]]: (N元词组, 频率)的列表
        """
        # 清理文本
        text = TextCleaner.clean_academic_text(text)
        
        # 分词
        words = text.split()
        
        # 生成N元词组
        ngrams = []
        for i in range(len(words) - n + 1):
            ngram = ' '.join(words[i:i+n])
            ngrams.append(ngram)
        
        # 计算词频
        ngram_freq = {}
        for ngram in ngrams:
            ngram_freq[ngram] = ngram_freq.get(ngram, 0) + 1
        
        # 过滤低频词组
        filtered_ngrams = [(ngram, freq) for ngram, freq in ngram_freq.items() if freq >= min_freq]
        
        # 按频率降序排序
        return sorted(filtered_ngrams, key=lambda x: x[1], reverse=True)
    
    @staticmethod
    def extract_academic_terms(text: str, academic_terms_dict: Dict[str, List[str]]) -> List[str]:
        """
        从文本中提取学术术语
        
        Args:
            text (str): 原始文本
            academic_terms_dict (Dict[str, List[str]]): 学术术语词典
            
        Returns:
            List[str]: 提取的术语列表
        """
        # 清理文本
        text = TextCleaner.clean_academic_text(text).lower()
        
        extracted_terms = []
        
        # 遍历领域术语词典
        for domain, terms in academic_terms_dict.items():
            for term in terms:
                term_lower = term.lower()
                # 使用正则表达式匹配完整词
                pattern = r'\b' + re.escape(term_lower) + r'\b'
                if re.search(pattern, text):
                    extracted_terms.append(term)
        
        return list(set(extracted_terms))
    
    @staticmethod
    def identify_citations(text: str) -> List[str]:
        """
        识别文本中的引用标记
        
        Args:
            text (str): 原始文本
            
        Returns:
            List[str]: 引用标记列表
        """
        # 匹配常见的引用格式
        # 1. [1], [2-4], [5,6,7]
        bracket_citations = re.findall(r'\[\d+(?:[-,]\d+)*\]', text)
        
        # 2. (Smith et al., 2020), (Smith and Jones, 2019)
        author_citations = re.findall(r'\([A-Za-z]+(?:\s+et\s+al\.|\s+and\s+[A-Za-z]+)?(?:,\s+\d{4})\)', text)
        
        # 3. Smith et al. (2020)
        inline_citations = re.findall(r'[A-Za-z]+(?:\s+et\s+al\.|\s+and\s+[A-Za-z]+)?\s+\(\d{4}\)', text)
        
        # 合并所有引用
        all_citations = bracket_citations + author_citations + inline_citations
        
        return all_citations
    
    @staticmethod
    def extract_references(text: str) -> List[str]:
        """
        从文本中提取参考文献
        
        Args:
            text (str): 原始文本
            
        Returns:
            List[str]: 参考文献列表
        """
        references = []
        
        # 查找参考文献部分
        ref_section_patterns = [
            r'References\s*\n([\s\S]+)',
            r'Bibliography\s*\n([\s\S]+)',
            r'参考文献\s*\n([\s\S]+)'
        ]
        
        for pattern in ref_section_patterns:
            match = re.search(pattern, text)
            if match:
                ref_section = match.group(1)
                
                # 尝试按编号分割参考文献
                numbered_refs = re.findall(r'^\s*\[\d+\](.+?)(?=^\s*\[\d+\]|\Z)', ref_section, re.MULTILINE | re.DOTALL)
                if numbered_refs:
                    references.extend([ref.strip() for ref in numbered_refs])
                    continue
                
                # 尝试按行分割参考文献
                line_refs = re.findall(r'^[^\n]+\(\d{4}\)[^\n]+', ref_section, re.MULTILINE)
                if line_refs:
                    references.extend([ref.strip() for ref in line_refs])
                    continue
                
                # 基本按段落分割
                para_refs = re.split(r'\n\s*\n', ref_section)
                references.extend([ref.strip() for ref in para_refs if ref.strip()])
        
        return references


class FormatConverter:
    """格式转换工具类，提供文本格式转换功能"""
    
    @staticmethod
    def text_to_markdown(text: str, doc_structure: Dict[str, Any] = None) -> str:
        """
        将纯文本转换为Markdown格式
        
        Args:
            text (str): 原始文本
            doc_structure (Dict[str, Any], optional): 文档结构信息
            
        Returns:
            str: Markdown格式文本
        """
        if not doc_structure:
            # 如果没有提供文档结构，尝试自动识别
            doc_structure = FormatConverter._detect_structure(text)
        
        # 分离标题、摘要和正文
        title = doc_structure.get('title', '')
        abstract = doc_structure.get('abstract', '')
        sections = doc_structure.get('sections', [])
        
        # 构建Markdown
        md_content = []
        
        # 添加标题
        if title:
            md_content.append(f"# {title}\n")
        
        # 添加摘要
        if abstract:
            md_content.append("## 摘要\n")
            md_content.append(f"{abstract}\n")
        
        # 添加正文部分
        for section in sections:
            section_title = section.get('title', '')
            section_content = section.get('content', '')
            
            if section_title:
                # 判断标题级别
                if section.get('level', 2) == 1:
                    md_content.append(f"# {section_title}\n")
                elif section.get('level', 2) == 2:
                    md_content.append(f"## {section_title}\n")
                else:
                    md_content.append(f"### {section_title}\n")
            
            if section_content:
                # 处理列表
                lines = section_content.split('\n')
                processed_lines = []
                
                for line in lines:
                    # 检测和转换列表项
                    if re.match(r'^\s*\d+\.\s', line):
                        # 有序列表
                        processed_lines.append(line)
                    elif re.match(r'^\s*[\-\*•]\s', line):
                        # 无序列表
                        processed_lines.append(line)
                    else:
                        processed_lines.append(line)
                
                section_content = '\n'.join(processed_lines)
                md_content.append(f"{section_content}\n")
        
        # 处理数学公式
        md_text = '\n'.join(md_content)
        md_text = FormatConverter._convert_math_formulas(md_text)
        
        # 转换引用和参考文献
        md_text = FormatConverter._convert_citations(md_text)
        
        return md_text
    
    @staticmethod
    def _detect_structure(text: str) -> Dict[str, Any]:
        """
        检测文本的结构
        
        Args:
            text (str): 原始文本
            
        Returns:
            Dict[str, Any]: 文档结构信息
        """
        # 分割文本为段落
        paragraphs = TextCleaner.extract_paragraphs(text)
        
        doc_structure = {
            'title': '',
            'abstract': '',
            'sections': []
        }
        
        # 提取标题（通常是第一个短段落）
        if paragraphs and len(paragraphs[0]) < 200:
            doc_structure['title'] = paragraphs[0]
            paragraphs = paragraphs[1:]
        
        # 提取摘要（通常在标题之后，以"Abstract"或"摘要"开头）
        for i, para in enumerate(paragraphs):
            if re.match(r'^(?:Abstract|摘要)[\s:]*', para, re.IGNORECASE):
                abstract_text = para
                # 可能的摘要继续
                if i + 1 < len(paragraphs) and len(paragraphs[i+1]) < 1000:
                    abstract_text += '\n\n' + paragraphs[i+1]
                
                doc_structure['abstract'] = re.sub(r'^(?:Abstract|摘要)[\s:]*', '', abstract_text)
                paragraphs = paragraphs[i+2:] if i + 1 < len(paragraphs) else paragraphs[i+1:]
                break
        
        # 处理剩余段落作为正文
        current_section = {'title': '', 'content': '', 'level': 2}
        
        for para in paragraphs:
            # 检测章节标题（通常较短，可能有数字前缀）
            if len(para) < 100 and re.match(r'^(?:\d+[\.\s]+)?[A-Z一-龥]', para):
                # 保存之前的章节
                if current_section['content']:
                    doc_structure['sections'].append(current_section)
                
                # 创建新章节
                level = 2
                if re.match(r'^\d+\.', para):  # 主章节
                    level = 2
                elif re.match(r'^\d+\.\d+', para):  # 子章节
                    level = 3
                
                current_section = {
                    'title': para,
                    'content': '',
                    'level': level
                }
            else:
                # 添加到当前章节内容
                if current_section['content']:
                    current_section['content'] += '\n\n' + para
                else:
                    current_section['content'] = para
        
        # 添加最后一个章节
        if current_section['content']:
            doc_structure['sections'].append(current_section)
        
        return doc_structure
    
    @staticmethod
    def _convert_math_formulas(text: str) -> str:
        """
        转换数学公式为Markdown格式
        
        Args:
            text (str): 原始文本
            
        Returns:
            str: 处理后的文本
        """
        # 转换行内公式，如 $E=mc^2$
        text = re.sub(r'(\$[^$\n]+\$)', r'\1', text)
        
        # 转换行间公式，如 $$\int_a^b f(x) dx$$
        def replace_block_formula(match):
            formula = match.group(1)
            return f'\n$$\n{formula}\n$$\n'
        
        text = re.sub(r'\$\$(.*?)\$\$', replace_block_formula, text, flags=re.DOTALL)
        
        return text
    
    @staticmethod
    def _convert_citations(text: str) -> str:
        """
        转换引用为Markdown格式
        
        Args:
            text (str): 原始文本
            
        Returns:
            str: 处理后的文本
        """
        # 转换引用
        citations = KeywordExtractor.identify_citations(text)
        
        for citation in citations:
            # 加粗引用
            text = text.replace(citation, f'**{citation}**')
        
        return text
    
    @staticmethod
    def extract_math_formulas(text: str) -> List[str]:
        """
        提取文本中的数学公式
        
        Args:
            text (str): 原始文本
            
        Returns:
            List[str]: 数学公式列表
        """
        # 提取行内公式，如 $E=mc^2$
        inline_formulas = re.findall(r'\$([^$\n]+)\$', text)
        
        # 提取行间公式，如 $$\int_a^b f(x) dx$$
        block_formulas = re.findall(r'\$\$(.*?)\$\$', text, re.DOTALL)
        
        return inline_formulas + block_formulas

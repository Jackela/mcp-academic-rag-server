"""
知识图谱处理器单元测试模块

测试知识图谱处理器的各项功能，包括实体提取、关系提取、概念提取等。
"""

import unittest
from unittest.mock import patch, MagicMock
import json

from processors.knowledge_graph_processor import KnowledgeGraphProcessor
from models.document import Document
from models.process_result import ProcessResult


class TestKnowledgeGraphProcessor(unittest.TestCase):
    """知识图谱处理器单元测试类"""
    
    def setUp(self):
        """测试前设置"""
        self.processor = KnowledgeGraphProcessor()
        
        # 创建测试文档
        self.test_document = Document("test_paper.pdf")
        self.test_document.file_type = "pdf"
        self.test_document.file_path = "test_paper.pdf"
        
        # 模拟OCR结果
        self.mock_ocr_result = {
            "text": self._get_sample_academic_text(),
            "pages": [{"page_num": 1, "text": self._get_sample_academic_text()}],
            "text_by_page": [self._get_sample_academic_text()]
        }
        
        # 模拟结构识别结果
        self.mock_structure_result = {
            "structure": {
                "title": "Machine Learning for Natural Language Processing",
                "abstract": "This paper presents a comprehensive study on machine learning approaches...",
                "sections": [
                    {
                        "title": "Introduction",
                        "content": "Natural language processing has witnessed significant advancements with deep learning techniques."
                    },
                    {
                        "title": "Methodology",
                        "content": "We propose a novel neural network architecture that combines CNN and RNN models."
                    }
                ]
            }
        }
        
        # 存储到测试文档
        self.test_document.store_content("ocr", self.mock_ocr_result)
        self.test_document.store_content("structure_recognition", self.mock_structure_result)
    
    def _get_sample_academic_text(self) -> str:
        """获取样本学术文本"""
        return """
        Machine Learning for Natural Language Processing
        
        Abstract
        This paper presents a comprehensive study on machine learning approaches for natural language processing tasks. 
        We compare convolutional neural networks, recurrent neural networks, and transformer models. 
        Our experiments demonstrate that transformer architectures achieve superior performance on text classification tasks.
        
        1. Introduction
        Natural language processing (NLP) has witnessed significant advancements with deep learning techniques. 
        Convolutional neural networks (CNNs) and recurrent neural networks (RNNs) have been widely used for NLP tasks.
        Smith et al. (2020) proposed a novel CNN architecture for text classification.
        
        2. Methodology
        We propose a neural network architecture that combines CNN and RNN models. 
        The proposed method uses feature extraction techniques and achieves 95% accuracy on benchmark datasets.
        Our approach improves the performance compared to traditional methods.
        
        3. Results
        Table 1: Performance Comparison
        Model     Accuracy  Precision
        CNN       0.85      0.82
        RNN       0.88      0.86
        Our Model 0.95      0.94
        
        Figure 1: System Architecture
        The figure shows the neural network architecture with multiple layers.
        """
    
    def test_initialization(self):
        """测试处理器初始化"""
        # 测试默认配置
        processor = KnowledgeGraphProcessor()
        self.assertEqual(processor.get_name(), "知识图谱处理器")
        self.assertEqual(processor.get_stage(), "knowledge_graph_extraction")
        self.assertTrue(processor.config['extract_entities'])
        self.assertTrue(processor.config['extract_relations'])
        self.assertTrue(processor.config['extract_concepts'])
        self.assertEqual(processor.config['min_entity_freq'], 2)
        self.assertEqual(processor.config['max_entities'], 100)
        self.assertEqual(processor.config['max_relations'], 50)
        
        # 测试自定义配置
        custom_config = {
            'extract_entities': False,
            'max_entities': 50,
            'min_entity_freq': 3
        }
        custom_processor = KnowledgeGraphProcessor(custom_config)
        self.assertFalse(custom_processor.config['extract_entities'])
        self.assertEqual(custom_processor.config['max_entities'], 50)
        self.assertEqual(custom_processor.config['min_entity_freq'], 3)
    
    def test_process_success(self):
        """测试成功处理文档"""
        result = self.processor.process(self.test_document)
        
        # 验证处理结果
        self.assertIsInstance(result, ProcessResult)
        self.assertTrue(result.is_successful())
        self.assertEqual(result.get_message(), "知识图谱提取完成")
        
        # 验证结果数据结构
        data = result.get_data()
        self.assertIn("knowledge_graph", data)
        self.assertIn("statistics", data)
        
        knowledge_graph = data["knowledge_graph"]
        self.assertIn("entities", knowledge_graph)
        self.assertIn("relations", knowledge_graph)
        self.assertIn("concepts", knowledge_graph)
        self.assertIn("metadata", knowledge_graph)
        
        # 验证文档状态更新
        self.assertTrue(self.test_document.get_metadata("has_knowledge_graph"))
        self.assertIsInstance(self.test_document.get_metadata("entity_count"), int)
        self.assertIsInstance(self.test_document.get_metadata("relation_count"), int)
    
    def test_process_no_structure_data(self):
        """测试没有结构数据时的处理"""
        # 创建没有结构数据的文档
        doc = Document("test_no_structure.pdf")
        doc.store_content("ocr", self.mock_ocr_result)
        
        result = self.processor.process(doc)
        
        # 验证错误处理
        self.assertFalse(result.is_successful())
        self.assertIn("无法获取文档结构", result.get_message())
    
    def test_extract_entities(self):
        """测试实体提取功能"""
        text = self._get_sample_academic_text()
        entities = self.processor._extract_entities(text)
        
        # 验证实体提取结果
        self.assertIsInstance(entities, dict)
        self.assertGreater(len(entities), 0)
        
        # 验证实体数据结构
        for entity_name, entity_info in entities.items():
            self.assertIsInstance(entity_name, str)
            self.assertIsInstance(entity_info, dict)
            self.assertIn('type', entity_info)
            self.assertIn('confidence', entity_info)
            self.assertIn('frequency', entity_info)
            self.assertIn('contexts', entity_info)
            
            # 验证数据类型
            self.assertIsInstance(entity_info['confidence'], (int, float))
            self.assertIsInstance(entity_info['frequency'], int)
            self.assertIsInstance(entity_info['contexts'], list)
            self.assertGreaterEqual(entity_info['confidence'], 0)
            self.assertLessEqual(entity_info['confidence'], 1)
            self.assertGreaterEqual(entity_info['frequency'], 1)
    
    def test_extract_pattern_based_entities(self):
        """测试基于模式的实体提取"""
        text = "John Smith and Mary Johnson developed a CNN algorithm for image processing. The RNN model achieved 90% accuracy."
        entities = self.processor._extract_pattern_based_entities(text)
        
        # 验证人名提取
        person_entities = [name for name, info in entities.items() if info['type'] == 'PERSON']
        self.assertGreater(len(person_entities), 0)
        
        # 验证技术术语提取
        tech_entities = [name for name, info in entities.items() if info['type'] == 'TECHNOLOGY']
        self.assertGreater(len(tech_entities), 0)
    
    def test_extract_term_based_entities(self):
        """测试基于术语词典的实体提取"""
        text = "We used machine learning algorithms and neural networks for natural language processing tasks."
        entities = self.processor._extract_term_based_entities(text)
        
        # 验证基于术语词典的实体提取
        self.assertGreater(len(entities), 0)
        
        # 验证特定术语被识别
        entity_names = [name.lower() for name in entities.keys()]
        self.assertTrue(any('machine learning' in name for name in entity_names) or 
                       any('algorithm' in name for name in entity_names))
    
    def test_extract_relations(self):
        """测试关系提取功能"""
        text = "CNN improves the accuracy. RNN is based on neural networks. The method uses feature extraction."
        entities = {
            'CNN': {'type': 'TECHNOLOGY', 'confidence': 0.9, 'frequency': 2},
            'RNN': {'type': 'TECHNOLOGY', 'confidence': 0.9, 'frequency': 2},
            'accuracy': {'type': 'METRIC', 'confidence': 0.8, 'frequency': 1},
            'neural networks': {'type': 'TECHNOLOGY', 'confidence': 0.9, 'frequency': 1},
            'feature extraction': {'type': 'TECHNOLOGY', 'confidence': 0.8, 'frequency': 1}
        }
        
        relations = self.processor._extract_relations(text, entities)
        
        # 验证关系提取结果
        self.assertIsInstance(relations, list)
        self.assertGreater(len(relations), 0)
        
        # 验证关系数据结构
        for relation in relations:
            self.assertIsInstance(relation, dict)
            self.assertIn('subject', relation)
            self.assertIn('predicate', relation)
            self.assertIn('object', relation)
            self.assertIn('confidence', relation)
            
            # 验证数据类型
            self.assertIsInstance(relation['subject'], str)
            self.assertIsInstance(relation['predicate'], str)
            self.assertIsInstance(relation['object'], str)
            self.assertIsInstance(relation['confidence'], (int, float))
            self.assertGreaterEqual(relation['confidence'], 0)
            self.assertLessEqual(relation['confidence'], 1)
    
    def test_extract_concepts(self):
        """测试概念提取功能"""
        text = "The concept of deep learning has revolutionized AI. Machine learning theory provides the foundation."
        concepts = self.processor._extract_concepts(text)
        
        # 验证概念提取结果
        self.assertIsInstance(concepts, list)
        self.assertGreaterEqual(len(concepts), 0)  # 可能没有匹配的概念模式
        
        # 如果有概念，验证数据结构
        for concept in concepts:
            self.assertIsInstance(concept, dict)
            self.assertIn('name', concept)
            self.assertIn('type', concept)
            self.assertIn('confidence', concept)
            
            # 验证数据类型
            self.assertIsInstance(concept['name'], str)
            self.assertEqual(concept['type'], 'CONCEPT')
            self.assertIsInstance(concept['confidence'], (int, float))
            self.assertGreaterEqual(concept['confidence'], 0)
            self.assertLessEqual(concept['confidence'], 1)
    
    def test_filter_entities(self):
        """测试实体过滤功能"""
        # 创建测试实体数据
        entities = {
            'high_conf_high_freq': {'confidence': 0.9, 'frequency': 5, 'type': 'TECH'},
            'high_conf_low_freq': {'confidence': 0.9, 'frequency': 1, 'type': 'TECH'},
            'low_conf_high_freq': {'confidence': 0.3, 'frequency': 5, 'type': 'TECH'},
            'low_conf_low_freq': {'confidence': 0.3, 'frequency': 1, 'type': 'TECH'}
        }
        
        # 添加更多实体来测试数量限制
        for i in range(150):
            entities[f'entity_{i}'] = {'confidence': 0.5, 'frequency': 2, 'type': 'TECH'}
        
        filtered = self.processor._filter_entities(entities)
        
        # 验证过滤结果
        self.assertLessEqual(len(filtered), self.processor.config['max_entities'])
        
        # 验证低频实体被过滤
        self.assertNotIn('high_conf_low_freq', filtered)
        self.assertNotIn('low_conf_low_freq', filtered)
    
    def test_deduplicate_relations(self):
        """测试关系去重功能"""
        relations = [
            {'subject': 'CNN', 'predicate': 'IMPROVES', 'object': 'accuracy', 'confidence': 0.8},
            {'subject': 'cnn', 'predicate': 'IMPROVES', 'object': 'Accuracy', 'confidence': 0.7},  # 重复（大小写不同）
            {'subject': 'RNN', 'predicate': 'USES', 'object': 'features', 'confidence': 0.6}
        ]
        
        unique_relations = self.processor._deduplicate_relations(relations)
        
        # 验证去重结果
        self.assertEqual(len(unique_relations), 2)  # 重复的关系应该被去除
    
    def test_get_entity_contexts(self):
        """测试获取实体上下文功能"""
        text = "CNN is a powerful model. CNN works well for images. CNN has many applications."
        contexts = self.processor._get_entity_contexts(text, "CNN", limit=2)
        
        # 验证上下文提取
        self.assertIsInstance(contexts, list)
        self.assertLessEqual(len(contexts), 2)  # 不超过限制
        self.assertGreater(len(contexts), 0)    # 应该找到上下文
        
        # 验证上下文内容
        for context in contexts:
            self.assertIsInstance(context, str)
            self.assertIn("CNN", context)
    
    def test_is_valid_ngram(self):
        """测试N-gram有效性验证"""
        # 有效的ngrams
        self.assertTrue(self.processor._is_valid_ngram("machine learning"))
        self.assertTrue(self.processor._is_valid_ngram("neural network architecture"))
        
        # 无效的ngrams
        self.assertFalse(self.processor._is_valid_ngram("a"))  # 太短
        self.assertFalse(self.processor._is_valid_ngram("123 456"))  # 纯数字
        self.assertFalse(self.processor._is_valid_ngram("the and or"))  # 停用词
    
    def test_is_valid_relation_entity(self):
        """测试关系实体有效性验证"""
        entities = {
            'CNN': {'type': 'TECHNOLOGY', 'confidence': 0.9},
            'accuracy': {'type': 'METRIC', 'confidence': 0.8}
        }
        
        # 有效实体
        self.assertTrue(self.processor._is_valid_relation_entity('CNN', entities))
        self.assertTrue(self.processor._is_valid_relation_entity('valid_term', entities))
        
        # 无效实体
        self.assertFalse(self.processor._is_valid_relation_entity('a', entities))  # 太短
        self.assertFalse(self.processor._is_valid_relation_entity('123', entities))  # 纯数字
    
    def test_infer_relation_type(self):
        """测试关系类型推断"""
        entity1_info = {'type': 'PERSON', 'confidence': 0.9}
        entity2_info = {'type': 'ORGANIZATION', 'confidence': 0.8}
        
        relation_type = self.processor._infer_relation_type(
            'John Smith', entity1_info, 'MIT', entity2_info
        )
        
        # 验证推断结果
        self.assertIsInstance(relation_type, str)
        self.assertIn(relation_type, ['AFFILIATED_WITH', 'RELATED_TO'])
    
    def test_detect_language(self):
        """测试语言检测功能"""
        # 英文文本
        english_text = "This is an English text about machine learning algorithms."
        lang = self.processor._detect_language(english_text)
        self.assertEqual(lang, "en")
        
        # 中文文本
        chinese_text = "这是一篇关于机器学习算法的中文文本。"
        lang = self.processor._detect_language(chinese_text)
        self.assertEqual(lang, "zh")
        
        # 混合文本（以英文为主）
        mixed_text = "This is mixed text 包含一些中文 but mostly English."
        lang = self.processor._detect_language(mixed_text)
        self.assertEqual(lang, "en")
    
    def test_preprocess_text(self):
        """测试文本预处理功能"""
        text = "This   is    a  text   with   extra    spaces   and   special#@#characters!!!"
        processed = self.processor._preprocess_text(text)
        
        # 验证预处理结果
        self.assertNotIn("   ", processed)  # 多余空格被移除
        self.assertNotIn("#", processed)    # 特殊字符被移除
        self.assertIn("This is a text", processed)  # 正常文本保留
    
    def test_calculate_statistics(self):
        """测试统计信息计算"""
        knowledge_graph = {
            "entities": {
                "CNN": {"type": "TECHNOLOGY", "confidence": 0.9, "frequency": 3},
                "RNN": {"type": "TECHNOLOGY", "confidence": 0.8, "frequency": 2},
                "accuracy": {"type": "METRIC", "confidence": 0.7, "frequency": 1}
            },
            "relations": [
                {"subject": "CNN", "predicate": "IMPROVES", "object": "accuracy", "confidence": 0.8},
                {"subject": "RNN", "predicate": "USES", "object": "features", "confidence": 0.7}
            ],
            "concepts": [
                {"name": "machine learning", "confidence": 0.9}
            ]
        }
        
        statistics = self.processor._calculate_statistics(knowledge_graph)
        
        # 验证统计结果
        self.assertEqual(statistics["total_entities"], 3)
        self.assertEqual(statistics["total_relations"], 2)
        self.assertEqual(statistics["total_concepts"], 1)
        self.assertEqual(statistics["entity_types"]["TECHNOLOGY"], 2)
        self.assertEqual(statistics["entity_types"]["METRIC"], 1)
        self.assertEqual(statistics["relation_types"]["IMPROVES"], 1)
        self.assertEqual(statistics["relation_types"]["USES"], 1)
        self.assertAlmostEqual(statistics["avg_entity_confidence"], 0.8, places=1)
        self.assertAlmostEqual(statistics["avg_relation_confidence"], 0.75, places=2)
    
    def test_get_main_entities(self):
        """测试获取主要实体功能"""
        entities = {
            "high_conf": {"confidence": 0.9, "frequency": 5},
            "medium_conf": {"confidence": 0.7, "frequency": 3},
            "low_conf": {"confidence": 0.5, "frequency": 2},
            "very_low_conf": {"confidence": 0.3, "frequency": 1}
        }
        
        main_entities = self.processor._get_main_entities(entities, limit=2)
        
        # 验证主要实体选择
        self.assertEqual(len(main_entities), 2)
        self.assertEqual(main_entities[0], "high_conf")  # 最高置信度的实体
        self.assertEqual(main_entities[1], "medium_conf")  # 第二高置信度的实体
    
    def test_configuration_effects(self):
        """测试配置参数对处理结果的影响"""
        # 测试禁用实体提取
        config_no_entities = {'extract_entities': False}
        processor_no_entities = KnowledgeGraphProcessor(config_no_entities)
        
        result = processor_no_entities.process(self.test_document)
        self.assertTrue(result.is_successful())
        
        kg_data = result.get_data()["knowledge_graph"]
        self.assertEqual(len(kg_data["entities"]), 0)
        
        # 测试禁用关系提取
        config_no_relations = {'extract_relations': False}
        processor_no_relations = KnowledgeGraphProcessor(config_no_relations)
        
        result = processor_no_relations.process(self.test_document)
        self.assertTrue(result.is_successful())
        
        kg_data = result.get_data()["knowledge_graph"]
        self.assertEqual(len(kg_data["relations"]), 0)
    
    def test_error_handling(self):
        """测试错误处理"""
        # 测试处理异常情况
        with patch.object(self.processor, '_extract_knowledge_graph', side_effect=Exception("Test error")):
            result = self.processor.process(self.test_document)
            
            self.assertFalse(result.is_successful())
            self.assertIn("知识图谱提取时发生错误", result.get_message())


if __name__ == "__main__":
    unittest.main()
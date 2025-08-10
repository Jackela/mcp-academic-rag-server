"""
知识图谱处理示例脚本

该脚本演示了如何在文档处理流水线中集成和使用知识图谱处理器，包括：
- 创建包含知识图谱处理器的处理流水线
- 处理学术文档并提取知识图谱
- 查看和分析提取的实体、关系和概念
- 导出知识图谱数据

用法：
    python knowledge_graph_example.py [config_path]
"""

import os
import sys
import json
import argparse
from pathlib import Path

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.pipeline import Pipeline
from models.document import Document
from processors.pre_processor import PreProcessor
from processors.ocr_processor import OCRProcessor
from processors.structure_processor import StructureProcessor
from processors.knowledge_graph_processor import KnowledgeGraphProcessor
from core.config_manager import ConfigManager


def create_knowledge_graph_pipeline(config_path: str) -> Pipeline:
    """
    创建包含知识图谱处理器的处理流水线
    
    Args:
        config_path (str): 配置文件路径
        
    Returns:
        Pipeline: 配置好的处理流水线
    """
    print("创建知识图谱处理流水线...")
    
    # 加载配置
    config_manager = ConfigManager(config_path)
    config = config_manager.get_config()
    
    # 创建流水线
    pipeline = Pipeline("KnowledgeGraphPipeline")
    
    # 添加预处理器
    if config.get("processors", {}).get("pre_processor", {}).get("enabled", False):
        pre_config = config["processors"]["pre_processor"]["config"]
        pre_processor = PreProcessor(pre_config)
        pipeline.add_processor(pre_processor)
        print("✓ 已添加预处理器")
    
    # 添加OCR处理器
    if config.get("processors", {}).get("ocr_processor", {}).get("enabled", False):
        ocr_config = config["processors"]["ocr_processor"]["config"]
        ocr_processor = OCRProcessor(ocr_config)
        pipeline.add_processor(ocr_processor)
        print("✓ 已添加OCR处理器")
    
    # 添加结构处理器
    if config.get("processors", {}).get("structure_processor", {}).get("enabled", False):
        structure_config = config["processors"]["structure_processor"]["config"]
        structure_processor = StructureProcessor(structure_config)
        pipeline.add_processor(structure_processor)
        print("✓ 已添加结构处理器")
    
    # 添加知识图谱处理器
    if config.get("processors", {}).get("knowledge_graph_processor", {}).get("enabled", False):
        kg_config = config["processors"]["knowledge_graph_processor"]["config"]
        kg_processor = KnowledgeGraphProcessor(kg_config)
        pipeline.add_processor(kg_processor)
        print("✓ 已添加知识图谱处理器")
    
    print(f"流水线创建完成，包含 {len(pipeline.get_processors())} 个处理器\n")
    return pipeline


def create_sample_document() -> Document:
    """
    创建样本学术文档
    
    Returns:
        Document: 样本文档对象
    """
    # 创建包含学术内容的样本文档
    sample_content = """
    Machine Learning Approaches for Natural Language Processing
    
    Abstract
    This paper presents a comprehensive study on machine learning approaches for natural language processing tasks. 
    We compare various neural network architectures including convolutional neural networks, recurrent neural networks, 
    and transformer models. Our experiments demonstrate that transformer-based models achieve superior performance 
    on text classification and sentiment analysis tasks.
    
    1. Introduction
    Natural language processing (NLP) has witnessed significant advancements with the introduction of deep learning 
    techniques. Convolutional neural networks (CNNs) and recurrent neural networks (RNNs) have been widely used 
    for various NLP tasks. Recently, transformer architectures have revolutionized the field by introducing 
    attention mechanisms that enable better understanding of contextual relationships.
    
    2. Related Work
    Smith et al. (2020) proposed a novel CNN architecture for text classification. Johnson and Brown (2021) 
    developed an improved RNN model for sentiment analysis. The transformer model introduced by Vaswani et al. (2017) 
    has become the foundation for many state-of-the-art NLP systems.
    
    3. Methodology
    Our approach combines multiple machine learning algorithms to create an ensemble model. We use feature extraction 
    techniques to process textual data and apply dimensionality reduction to improve computational efficiency. 
    The proposed method achieves 95% accuracy on the benchmark dataset.
    
    Table 1: Performance Comparison
    Model       Accuracy    Precision   Recall
    CNN         0.85        0.82        0.87
    RNN         0.88        0.86        0.89
    Transformer 0.95        0.94        0.96
    
    Figure 1: Architecture Diagram
    The figure shows the overall system architecture with three main components: feature extraction, 
    model training, and evaluation.
    
    4. Conclusion
    In this work, we demonstrated that transformer-based approaches outperform traditional CNN and RNN models 
    for NLP tasks. The attention mechanism enables better capture of long-range dependencies in text.
    
    References
    [1] Smith, J., et al. (2020). Advanced CNN architectures for text processing. Journal of Machine Learning.
    [2] Johnson, A., Brown, B. (2021). Improved RNN models for sentiment analysis. Conference on NLP.
    [3] Vaswani, A., et al. (2017). Attention is all you need. NIPS.
    """
    
    # 创建文档对象
    doc = Document("sample_academic_paper.txt")
    doc.file_type = "text"
    doc.file_path = "sample_academic_paper.txt"
    
    # 模拟OCR处理结果
    ocr_result = {
        "text": sample_content,
        "pages": [{"page_num": 1, "text": sample_content}],
        "text_by_page": [sample_content]
    }
    doc.store_content("ocr", ocr_result)
    
    return doc


def analyze_knowledge_graph(kg_data: dict) -> None:
    """
    分析和展示知识图谱数据
    
    Args:
        kg_data (dict): 知识图谱数据
    """
    print("知识图谱分析结果")
    print("=" * 60)
    
    knowledge_graph = kg_data.get("knowledge_graph", {})
    statistics = kg_data.get("statistics", {})
    
    # 展示统计信息
    print("\n📊 统计信息:")
    print(f"  实体总数: {statistics.get('total_entities', 0)}")
    print(f"  关系总数: {statistics.get('total_relations', 0)}")
    print(f"  概念总数: {statistics.get('total_concepts', 0)}")
    print(f"  平均实体置信度: {statistics.get('avg_entity_confidence', 0):.3f}")
    print(f"  平均关系置信度: {statistics.get('avg_relation_confidence', 0):.3f}")
    
    # 展示实体类型分布
    entity_types = statistics.get('entity_types', {})
    if entity_types:
        print("\n🏷️  实体类型分布:")
        for entity_type, count in entity_types.items():
            print(f"  {entity_type}: {count}")
    
    # 展示关系类型分布
    relation_types = statistics.get('relation_types', {})
    if relation_types:
        print("\n🔗 关系类型分布:")
        for relation_type, count in relation_types.items():
            print(f"  {relation_type}: {count}")
    
    # 展示主要实体
    entities = knowledge_graph.get("entities", {})
    if entities:
        print("\n🎯 主要实体 (Top 10):")
        sorted_entities = sorted(
            entities.items(),
            key=lambda x: (x[1].get('confidence', 0), x[1].get('frequency', 0)),
            reverse=True
        )
        
        for i, (entity_name, entity_info) in enumerate(sorted_entities[:10], 1):
            entity_type = entity_info.get('type', 'UNKNOWN')
            confidence = entity_info.get('confidence', 0)
            frequency = entity_info.get('frequency', 0)
            print(f"  {i:2d}. {entity_name} ({entity_type}) - 置信度: {confidence:.3f}, 频率: {frequency}")
    
    # 展示主要关系
    relations = knowledge_graph.get("relations", [])
    if relations:
        print("\n🔄 主要关系 (Top 10):")
        sorted_relations = sorted(relations, key=lambda x: x.get('confidence', 0), reverse=True)
        
        for i, relation in enumerate(sorted_relations[:10], 1):
            subject = relation.get('subject', '')
            predicate = relation.get('predicate', '')
            obj = relation.get('object', '')
            confidence = relation.get('confidence', 0)
            print(f"  {i:2d}. {subject} --[{predicate}]--> {obj} (置信度: {confidence:.3f})")
    
    # 展示概念
    concepts = knowledge_graph.get("concepts", [])
    if concepts:
        print("\n💡 提取的概念 (Top 5):")
        sorted_concepts = sorted(concepts, key=lambda x: x.get('confidence', 0), reverse=True)
        
        for i, concept in enumerate(sorted_concepts[:5], 1):
            concept_name = concept.get('name', '')
            confidence = concept.get('confidence', 0)
            print(f"  {i}. {concept_name} (置信度: {confidence:.3f})")


def export_knowledge_graph(kg_data: dict, output_path: str) -> None:
    """
    导出知识图谱数据到文件
    
    Args:
        kg_data (dict): 知识图谱数据
        output_path (str): 输出文件路径
    """
    print(f"\n📄 导出知识图谱数据到: {output_path}")
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(kg_data, f, ensure_ascii=False, indent=2)
        print("✓ 导出成功")
    except Exception as e:
        print(f"✗ 导出失败: {e}")


def generate_cypher_queries(kg_data: dict) -> List[str]:
    """
    生成用于图数据库的Cypher查询语句
    
    Args:
        kg_data (dict): 知识图谱数据
        
    Returns:
        List[str]: Cypher查询语句列表
    """
    queries = []
    knowledge_graph = kg_data.get("knowledge_graph", {})
    
    # 创建实体节点
    entities = knowledge_graph.get("entities", {})
    for entity_name, entity_info in entities.items():
        entity_type = entity_info.get('type', 'Entity')
        confidence = entity_info.get('confidence', 0)
        frequency = entity_info.get('frequency', 0)
        
        # 转义实体名称中的特殊字符
        escaped_name = entity_name.replace("'", "\\'").replace('"', '\\"')
        
        query = f"""CREATE (:{entity_type} {{
    name: '{escaped_name}',
    confidence: {confidence},
    frequency: {frequency}
}})"""
        queries.append(query)
    
    # 创建关系
    relations = knowledge_graph.get("relations", [])
    for relation in relations:
        subject = relation.get('subject', '').replace("'", "\\'")
        predicate = relation.get('predicate', 'RELATED_TO')
        obj = relation.get('object', '').replace("'", "\\'")
        confidence = relation.get('confidence', 0)
        
        query = f"""MATCH (a {{name: '{subject}'}}), (b {{name: '{obj}'}})
CREATE (a)-[:{predicate} {{confidence: {confidence}}}]->(b)"""
        queries.append(query)
    
    return queries


def run_knowledge_graph_example(config_path: str) -> None:
    """
    运行知识图谱处理示例
    
    Args:
        config_path (str): 配置文件路径
    """
    print("知识图谱处理示例")
    print("=" * 80 + "\n")
    
    try:
        # 1. 创建处理流水线
        pipeline = create_knowledge_graph_pipeline(config_path)
        
        # 2. 创建样本文档
        print("创建样本学术文档...")
        document = create_sample_document()
        print("✓ 样本文档创建完成\n")
        
        # 3. 模拟结构处理结果（通常由StructureProcessor生成）
        print("模拟文档结构处理...")
        structure_result = {
            "structure": {
                "title": "Machine Learning Approaches for Natural Language Processing",
                "abstract": "This paper presents a comprehensive study on machine learning approaches for natural language processing tasks...",
                "sections": [
                    {
                        "title": "Introduction",
                        "content": "Natural language processing (NLP) has witnessed significant advancements..."
                    },
                    {
                        "title": "Methodology", 
                        "content": "Our approach combines multiple machine learning algorithms..."
                    }
                ]
            }
        }
        document.store_content("structure_recognition", structure_result)
        print("✓ 结构处理完成\n")
        
        # 4. 运行知识图谱处理器
        print("执行知识图谱提取...")
        kg_processor = KnowledgeGraphProcessor()
        result = kg_processor.process(document)
        
        if result.is_successful():
            print("✓ 知识图谱提取成功\n")
            
            # 5. 分析结果
            kg_data = result.get_data()
            analyze_knowledge_graph(kg_data)
            
            # 6. 导出结果
            output_dir = os.path.join(os.path.dirname(__file__), "output")
            os.makedirs(output_dir, exist_ok=True)
            
            output_path = os.path.join(output_dir, "knowledge_graph.json")
            export_knowledge_graph(kg_data, output_path)
            
            # 7. 生成Cypher查询示例
            print("\n🔍 生成的Cypher查询示例 (前5个):")
            cypher_queries = generate_cypher_queries(kg_data)
            for i, query in enumerate(cypher_queries[:5], 1):
                print(f"\n查询 {i}:")
                print(query)
            
            cypher_path = os.path.join(output_dir, "cypher_queries.cql")
            with open(cypher_path, 'w', encoding='utf-8') as f:
                f.write('\n\n'.join(cypher_queries))
            print(f"\n✓ 完整的Cypher查询已保存到: {cypher_path}")
            
        else:
            print(f"✗ 知识图谱提取失败: {result.get_message()}")
    
    except Exception as e:
        print(f"✗ 示例运行失败: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        print("\n" + "=" * 80)
        print("知识图谱处理示例完成")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="知识图谱处理示例脚本")
    parser.add_argument("config_path", nargs="?", default="./config/config.json", 
                        help="配置文件路径")
    args = parser.parse_args()
    
    run_knowledge_graph_example(args.config_path)


if __name__ == "__main__":
    main()
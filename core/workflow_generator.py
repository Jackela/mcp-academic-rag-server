"""
工作流生成器核心引擎

基于PRD和需求描述生成完整的实施工作流。
"""

import logging
import asyncio
import re
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime, timedelta
from pathlib import Path

from .workflow_models import (
    Workflow, WorkflowStrategy, OutputFormat, PersonaType, Priority, ComplexityLevel,
    PRDStructure, Requirement, AcceptanceCriteria, Dependency, Risk, 
    WorkflowPhase, WorkflowMilestone, WorkflowStep, ParallelStream, CriticalPath,
    EffortEstimation, ComplexityAnalysis, RequirementCategories, MCPResults, WorkflowOptions
)
from .workflow_effort_calculator import EffortCalculator, PhaseEffortDistributor


class PRDParser:
    """PRD文档解析器"""
    
    def __init__(self):
        self.logger = logging.getLogger("PRDParser")
    
    async def parse_document(self, file_path: str) -> PRDStructure:
        """解析PRD文档文件"""
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"PRD文档不存在: {file_path}")
            
            content = file_path.read_text(encoding='utf-8')
            return self.parse_content(content, file_path.name)
            
        except Exception as e:
            self.logger.error(f"解析PRD文档失败: {str(e)}")
            raise
    
    def parse_content(self, content: str, source_name: str = "text_input") -> PRDStructure:
        """解析PRD文档内容"""
        try:
            prd = PRDStructure()
            
            # 提取标题
            title_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
            if title_match:
                prd.title = title_match.group(1).strip()
            else:
                prd.title = source_name
            
            # 提取概述
            overview_match = re.search(r'(?:## 概述|## Overview|## 简介)(.*?)(?=##|$)', content, re.DOTALL | re.IGNORECASE)
            if overview_match:
                prd.overview = overview_match.group(1).strip()
            
            # 提取目标
            objectives = self._extract_list_items(content, ['目标', 'objectives', '项目目标'])
            prd.objectives = objectives
            
            # 提取需求
            requirements = self._extract_requirements(content)
            prd.requirements = requirements
            
            # 提取验收标准
            acceptance_criteria = self._extract_acceptance_criteria(content, requirements)
            prd.acceptance_criteria = acceptance_criteria
            
            # 提取约束
            constraints = self._extract_list_items(content, ['约束', 'constraints', '限制'])
            prd.constraints = constraints
            
            # 提取假设
            assumptions = self._extract_list_items(content, ['假设', 'assumptions'])
            prd.assumptions = assumptions
            
            # 提取成功指标
            success_metrics = self._extract_list_items(content, ['成功指标', 'success metrics', 'kpi'])
            prd.success_metrics = success_metrics
            
            prd.date_created = datetime.now()
            prd.metadata = {'source': source_name, 'parsing_method': 'regex'}
            
            self.logger.info(f"成功解析PRD: {len(requirements)}个需求, {len(acceptance_criteria)}个验收标准")
            return prd
            
        except Exception as e:
            self.logger.error(f"解析PRD内容失败: {str(e)}")
            raise
    
    def parse_text_description(self, description: str) -> PRDStructure:
        """解析简单文本描述为PRD结构"""
        try:
            prd = PRDStructure()
            prd.title = "文本描述需求"
            prd.overview = description
            prd.date_created = datetime.now()
            
            # 从描述中提取需求
            sentences = [s.strip() for s in description.split('.') if s.strip()]
            
            requirements = []
            for i, sentence in enumerate(sentences[:5]):  # 最多5个需求
                if len(sentence) > 10:  # 过滤太短的句子
                    req = Requirement(
                        title=f"需求 {i+1}",
                        description=sentence,
                        priority=Priority.MEDIUM,
                        complexity=self._estimate_sentence_complexity(sentence)
                    )
                    requirements.append(req)
            
            prd.requirements = requirements
            prd.metadata = {'source': 'text_description', 'auto_generated': True}
            
            self.logger.info(f"从文本描述生成 {len(requirements)} 个需求")
            return prd
            
        except Exception as e:
            self.logger.error(f"解析文本描述失败: {str(e)}")
            raise
    
    def _extract_requirements(self, content: str) -> List[Requirement]:
        """提取功能需求"""
        requirements = []
        
        # 查找需求章节
        sections = ['需求', 'requirements', '功能需求', 'functional requirements']
        for section in sections:
            pattern = rf'(?:## {section}|### {section})(.*?)(?=##|$)'
            match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
            if match:
                section_content = match.group(1)
                break
        else:
            # 如果没找到专门的需求章节，从整个文档提取
            section_content = content
        
        # 提取编号列表项
        list_items = re.findall(r'(?:^|\n)(?:\d+\.|\-|\*)\s+(.+?)(?=\n(?:\d+\.|\-|\*)|$)', 
                               section_content, re.MULTILINE | re.DOTALL)
        
        for i, item in enumerate(list_items):
            if len(item.strip()) > 5:  # 过滤太短的项目
                req = Requirement(
                    title=f"需求 {i+1}",
                    description=item.strip(),
                    priority=self._extract_priority(item),
                    complexity=self._estimate_requirement_complexity(item)
                )
                requirements.append(req)
        
        # 如果没有找到列表项，尝试提取段落作为需求
        if not requirements:
            paragraphs = [p.strip() for p in section_content.split('\n\n') if p.strip()]
            for i, para in enumerate(paragraphs[:3]):  # 最多3个段落
                if len(para) > 20:
                    req = Requirement(
                        title=f"需求 {i+1}",
                        description=para,
                        priority=Priority.MEDIUM
                    )
                    requirements.append(req)
        
        return requirements
    
    def _extract_acceptance_criteria(self, content: str, requirements: List[Requirement]) -> List[AcceptanceCriteria]:
        """提取验收标准"""
        criteria = []
        
        # 查找验收标准章节
        sections = ['验收标准', 'acceptance criteria', '验收条件']
        for section in sections:
            pattern = rf'(?:## {section}|### {section})(.*?)(?=##|$)'
            match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
            if match:
                section_content = match.group(1)
                
                # 提取列表项作为验收标准
                list_items = re.findall(r'(?:^|\n)(?:\-|\*)\s+(.+?)(?=\n(?:\-|\*)|$)', 
                                       section_content, re.MULTILINE | re.DOTALL)
                
                for item in list_items:
                    if len(item.strip()) > 5:
                        ac = AcceptanceCriteria(
                            description=item.strip(),
                            is_testable=self._is_testable(item),
                            test_method=self._determine_test_method(item)
                        )
                        # 尝试关联到相关需求
                        if requirements:
                            ac.requirement_id = requirements[0].id  # 简化关联逻辑
                        criteria.append(ac)
                break
        
        return criteria
    
    def _extract_list_items(self, content: str, keywords: List[str]) -> List[str]:
        """提取指定关键词的列表项"""
        items = []
        
        for keyword in keywords:
            pattern = rf'(?:## {keyword}|### {keyword})(.*?)(?=##|$)'
            match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
            if match:
                section_content = match.group(1)
                
                list_items = re.findall(r'(?:^|\n)(?:\d+\.|\-|\*)\s+(.+?)(?=\n(?:\d+\.|\-|\*)|$)', 
                                       section_content, re.MULTILINE | re.DOTALL)
                
                items.extend([item.strip() for item in list_items if len(item.strip()) > 3])
                break
        
        return items
    
    def _extract_priority(self, text: str) -> Priority:
        """从文本中提取优先级"""
        text_lower = text.lower()
        if any(word in text_lower for word in ['高优先级', 'high', 'critical', '紧急', '关键']):
            return Priority.HIGH
        elif any(word in text_lower for word in ['低优先级', 'low', '可选', 'optional']):
            return Priority.LOW
        else:
            return Priority.MEDIUM
    
    def _estimate_requirement_complexity(self, text: str) -> ComplexityLevel:
        """估算需求复杂度"""
        text_lower = text.lower()
        complexity_indicators = {
            'simple': ['简单', '基础', 'simple', 'basic', '显示', '展示'],
            'complex': ['复杂', '集成', '算法', '安全', '性能', 'complex', 'integration', 'algorithm', 'security', 'performance'],
            'enterprise': ['企业级', '分布式', '高可用', '大规模', 'enterprise', 'distributed', 'scalable']
        }
        
        if any(word in text_lower for word in complexity_indicators['enterprise']):
            return ComplexityLevel.ENTERPRISE
        elif any(word in text_lower for word in complexity_indicators['complex']):
            return ComplexityLevel.COMPLEX
        elif any(word in text_lower for word in complexity_indicators['simple']):
            return ComplexityLevel.SIMPLE
        else:
            return ComplexityLevel.MEDIUM
    
    def _estimate_sentence_complexity(self, sentence: str) -> ComplexityLevel:
        """基于句子长度和内容估算复杂度"""
        if len(sentence) < 20:
            return ComplexityLevel.SIMPLE
        elif len(sentence) > 100:
            return ComplexityLevel.COMPLEX
        else:
            return ComplexityLevel.MEDIUM
    
    def _is_testable(self, criteria: str) -> bool:
        """判断验收标准是否可测试"""
        testable_indicators = ['能够', '可以', '应该', '验证', '测试', 'should', 'can', 'able to', 'verify']
        return any(indicator in criteria.lower() for indicator in testable_indicators)
    
    def _determine_test_method(self, criteria: str) -> str:
        """确定测试方法"""
        if any(word in criteria.lower() for word in ['自动', 'automatic', 'api', '接口']):
            return 'automated'
        elif any(word in criteria.lower() for word in ['集成', 'integration', '端到端']):
            return 'integration'
        else:
            return 'manual'


class RequirementAnalyzer:
    """需求分析器"""
    
    def __init__(self):
        self.logger = logging.getLogger("RequirementAnalyzer")
    
    def analyze_complexity(self, requirements: List[Requirement]) -> ComplexityAnalysis:
        """分析需求复杂度"""
        try:
            if not requirements:
                return ComplexityAnalysis()
            
            # 统计各复杂度级别的需求数量
            complexity_counts = {level: 0 for level in ComplexityLevel}
            for req in requirements:
                complexity_counts[req.complexity] += 1
            
            total_reqs = len(requirements)
            
            # 计算加权平均复杂度
            complexity_weights = {
                ComplexityLevel.SIMPLE: 0.2,
                ComplexityLevel.MEDIUM: 0.5,
                ComplexityLevel.COMPLEX: 0.8,
                ComplexityLevel.ENTERPRISE: 1.0
            }
            
            weighted_sum = sum(complexity_counts[level] * complexity_weights[level] 
                             for level in ComplexityLevel)
            average_complexity = weighted_sum / total_reqs if total_reqs > 0 else 0.5
            
            # 确定整体复杂度级别
            if average_complexity < 0.3:
                overall_complexity = ComplexityLevel.SIMPLE
            elif average_complexity < 0.6:
                overall_complexity = ComplexityLevel.MEDIUM
            elif average_complexity < 0.9:
                overall_complexity = ComplexityLevel.COMPLEX
            else:
                overall_complexity = ComplexityLevel.ENTERPRISE
            
            # 分析复杂度因素
            complexity_factors = []
            if complexity_counts[ComplexityLevel.ENTERPRISE] > 0:
                complexity_factors.append("包含企业级需求")
            if complexity_counts[ComplexityLevel.COMPLEX] > total_reqs * 0.5:
                complexity_factors.append("复杂需求占比过半")
            if total_reqs > 20:
                complexity_factors.append("需求数量较多")
            
            # 识别简化机会
            simplification_opportunities = []
            if complexity_counts[ComplexityLevel.COMPLEX] > 0:
                simplification_opportunities.append("考虑将复杂需求分解为更小的需求")
            if total_reqs > 15:
                simplification_opportunities.append("考虑采用MVP策略，分阶段实施")
            
            analysis = ComplexityAnalysis(
                overall_complexity=overall_complexity,
                technical_complexity=average_complexity,
                integration_complexity=self._analyze_integration_complexity(requirements),
                ui_complexity=self._analyze_ui_complexity(requirements),
                business_logic_complexity=self._analyze_business_logic_complexity(requirements),
                data_complexity=self._analyze_data_complexity(requirements),
                complexity_factors=complexity_factors,
                simplification_opportunities=simplification_opportunities
            )
            
            self.logger.info(f"复杂度分析完成: {overall_complexity.value}, 技术复杂度: {average_complexity:.2f}")
            return analysis
            
        except Exception as e:
            self.logger.error(f"复杂度分析失败: {str(e)}")
            return ComplexityAnalysis()
    
    def categorize_requirements(self, requirements: List[Requirement]) -> RequirementCategories:
        """需求分类"""
        try:
            categories = RequirementCategories()
            
            for req in requirements:
                desc_lower = req.description.lower()
                
                # 基于描述内容进行分类
                if req.type == 'functional' or self._is_functional_requirement(desc_lower):
                    categories.functional_requirements.append(req)
                elif req.type == 'non-functional' or self._is_non_functional_requirement(desc_lower):
                    categories.non_functional_requirements.append(req)
                elif req.type == 'constraint':
                    categories.constraint_requirements.append(req)
                
                # 进一步细分
                if self._is_ui_requirement(desc_lower):
                    categories.ui_requirements.append(req)
                if self._is_api_requirement(desc_lower):
                    categories.api_requirements.append(req)
                if self._is_data_requirement(desc_lower):
                    categories.data_requirements.append(req)
                if self._is_security_requirement(desc_lower):
                    categories.security_requirements.append(req)
                if self._is_performance_requirement(desc_lower):
                    categories.performance_requirements.append(req)
            
            self.logger.info(f"需求分类完成: 功能性({len(categories.functional_requirements)}),"
                           f"非功能性({len(categories.non_functional_requirements)}),"
                           f"约束性({len(categories.constraint_requirements)})")
            return categories
            
        except Exception as e:
            self.logger.error(f"需求分类失败: {str(e)}")
            return RequirementCategories()
    
    def estimate_effort(self, requirements: List[Requirement]) -> EffortEstimation:
        """工作量评估"""
        try:
            if not requirements:
                return EffortEstimation()
            
            # 基于复杂度的基础工作量估算
            complexity_hours = {
                ComplexityLevel.SIMPLE: 4,      # 4小时
                ComplexityLevel.MEDIUM: 12,     # 1.5天
                ComplexityLevel.COMPLEX: 32,    # 4天
                ComplexityLevel.ENTERPRISE: 80  # 10天
            }
            
            total_hours = 0
            breakdown_by_persona = {}
            
            for req in requirements:
                base_hours = complexity_hours.get(req.complexity, 12)
                
                # 根据需求类型调整估算
                if self._is_ui_requirement(req.description.lower()):
                    hours = base_hours * 1.2  # UI需求增加20%
                    persona = PersonaType.FRONTEND
                elif self._is_api_requirement(req.description.lower()):
                    hours = base_hours * 1.1  # API需求增加10%
                    persona = PersonaType.BACKEND
                elif self._is_security_requirement(req.description.lower()):
                    hours = base_hours * 1.5  # 安全需求增加50%
                    persona = PersonaType.SECURITY
                else:
                    hours = base_hours
                    persona = PersonaType.BACKEND  # 默认后端
                
                total_hours += hours
                
                if persona not in breakdown_by_persona:
                    breakdown_by_persona[persona] = 0
                breakdown_by_persona[persona] += hours
            
            # 添加20%的缓冲
            buffered_hours = int(total_hours * 1.2)
            
            estimation = EffortEstimation(
                total_hours=buffered_hours,
                breakdown_by_persona=breakdown_by_persona,
                confidence_level=0.7,  # 中等置信度
                estimation_method="complexity_based",
                buffer_percentage=0.2
            )
            
            self.logger.info(f"工作量评估完成: {buffered_hours}小时 (包含20%缓冲)")
            return estimation
            
        except Exception as e:
            self.logger.error(f"工作量评估失败: {str(e)}")
            return EffortEstimation()
    
    def _analyze_integration_complexity(self, requirements: List[Requirement]) -> float:
        """分析集成复杂度"""
        integration_keywords = ['集成', '对接', '同步', 'integration', 'api', '第三方']
        integration_reqs = [req for req in requirements 
                           if any(keyword in req.description.lower() for keyword in integration_keywords)]
        return min(len(integration_reqs) / len(requirements) * 2, 1.0) if requirements else 0.0
    
    def _analyze_ui_complexity(self, requirements: List[Requirement]) -> float:
        """分析UI复杂度"""
        ui_keywords = ['界面', '页面', '组件', 'ui', '前端', '交互', '用户体验']
        ui_reqs = [req for req in requirements 
                  if any(keyword in req.description.lower() for keyword in ui_keywords)]
        return min(len(ui_reqs) / len(requirements) * 1.5, 1.0) if requirements else 0.0
    
    def _analyze_business_logic_complexity(self, requirements: List[Requirement]) -> float:
        """分析业务逻辑复杂度"""
        logic_keywords = ['业务逻辑', '规则', '流程', '算法', 'logic', 'rule', 'process']
        logic_reqs = [req for req in requirements 
                     if any(keyword in req.description.lower() for keyword in logic_keywords)]
        return min(len(logic_reqs) / len(requirements) * 1.8, 1.0) if requirements else 0.0
    
    def _analyze_data_complexity(self, requirements: List[Requirement]) -> float:
        """分析数据复杂度"""
        data_keywords = ['数据', '存储', '数据库', 'data', 'database', '模型']
        data_reqs = [req for req in requirements 
                    if any(keyword in req.description.lower() for keyword in data_keywords)]
        return min(len(data_reqs) / len(requirements) * 1.3, 1.0) if requirements else 0.0
    
    def _is_functional_requirement(self, desc: str) -> bool:
        """判断是否为功能性需求"""
        functional_keywords = ['实现', '功能', '能够', '可以', '支持', 'feature', 'function']
        return any(keyword in desc for keyword in functional_keywords)
    
    def _is_non_functional_requirement(self, desc: str) -> bool:
        """判断是否为非功能性需求"""
        non_functional_keywords = ['性能', '安全', '可用性', '扩展性', 'performance', 'security', 'scalability']
        return any(keyword in desc for keyword in non_functional_keywords)
    
    def _is_ui_requirement(self, desc: str) -> bool:
        """判断是否为UI需求"""
        ui_keywords = ['界面', '页面', '组件', 'ui', '前端', '用户界面', '交互']
        return any(keyword in desc for keyword in ui_keywords)
    
    def _is_api_requirement(self, desc: str) -> bool:
        """判断是否为API需求"""
        api_keywords = ['api', '接口', '服务', '端点', 'endpoint', 'service']
        return any(keyword in desc for keyword in api_keywords)
    
    def _is_data_requirement(self, desc: str) -> bool:
        """判断是否为数据需求"""
        data_keywords = ['数据', '存储', '数据库', '模型', 'data', 'database', 'model']
        return any(keyword in desc for keyword in data_keywords)
    
    def _is_security_requirement(self, desc: str) -> bool:
        """判断是否为安全需求"""
        security_keywords = ['安全', '认证', '授权', '加密', 'security', 'auth', 'encryption']
        return any(keyword in desc for keyword in security_keywords)
    
    def _is_performance_requirement(self, desc: str) -> bool:
        """判断是否为性能需求"""
        performance_keywords = ['性能', '速度', '响应时间', '吞吐量', 'performance', 'speed', 'latency']
        return any(keyword in desc for keyword in performance_keywords)


class WorkflowGenerator:
    """工作流生成器核心引擎"""
    
    def __init__(self):
        self.logger = logging.getLogger("WorkflowGenerator")
        self.prd_parser = PRDParser()
        self.requirement_analyzer = RequirementAnalyzer()
        self.effort_calculator = EffortCalculator()
        self.phase_distributor = PhaseEffortDistributor()
    
    async def generate_workflow(
        self, 
        input_source: str,
        options: WorkflowOptions = None
    ) -> Workflow:
        """生成完整工作流"""
        try:
            if options is None:
                options = WorkflowOptions()
            
            self.logger.info(f"开始生成工作流: {options.strategy.value} 策略")
            
            # 1. 解析输入源
            prd_structure = await self._parse_input_source(input_source)
            
            # 2. 分析需求
            complexity_analysis = self.requirement_analyzer.analyze_complexity(prd_structure.requirements)
            requirement_categories = self.requirement_analyzer.categorize_requirements(prd_structure.requirements)
            effort_estimation = self.requirement_analyzer.estimate_effort(prd_structure.requirements)
            
            # 3. 创建基础工作流对象
            workflow = Workflow(
                name=prd_structure.title or "工作流",
                description=prd_structure.overview or "自动生成的工作流",
                strategy=options.strategy,
                prd_structure=prd_structure,
                requirements=prd_structure.requirements,
                complexity_analysis=complexity_analysis,
                requirement_categories=requirement_categories,
                effort_estimation=effort_estimation
            )
            
            # 4. 根据策略生成工作流阶段
            phases = await self._generate_phases(workflow, options)
            workflow.phases = phases
            
            # 5. 分析依赖关系
            if options.include_dependencies:
                dependencies = await self._analyze_dependencies(workflow)
                workflow.dependencies = dependencies
            
            # 6. 评估风险
            if options.include_risks:
                risks = await self._assess_risks(workflow)
                workflow.risks = risks
            
            # 7. 识别并行流
            if options.enable_parallel_analysis:
                parallel_streams = await self._identify_parallel_streams(workflow)
                workflow.parallel_streams = parallel_streams
                
                critical_path = await self._calculate_critical_path(workflow)
                workflow.critical_path = critical_path
            
            # 8. 激活专家人格
            activated_personas = await self._activate_personas(workflow, options)
            workflow.activated_personas = activated_personas
            
            # 9. MCP集成（如果启用）
            if options.enable_mcp_integration:
                mcp_results = await self._integrate_mcp_services(workflow, options)
                workflow.mcp_results = mcp_results
            
            workflow.updated_at = datetime.now()
            
            self.logger.info(f"工作流生成完成: {len(phases)}个阶段, "
                           f"{workflow.get_total_effort()}小时, "
                           f"复杂度: {complexity_analysis.overall_complexity.value}")
            
            return workflow
            
        except Exception as e:
            self.logger.error(f"生成工作流失败: {str(e)}")
            raise
    
    async def _parse_input_source(self, input_source: str) -> PRDStructure:
        """解析输入源"""
        try:
            # 检查是否是文件路径
            if Path(input_source).exists():
                return await self.prd_parser.parse_document(input_source)
            else:
                # 作为文本描述处理
                return self.prd_parser.parse_text_description(input_source)
                
        except Exception as e:
            self.logger.error(f"解析输入源失败: {str(e)}")
            raise
    
    async def _generate_phases(self, workflow: Workflow, options: WorkflowOptions) -> List[WorkflowPhase]:
        """根据策略生成工作流阶段"""
        try:
            if options.strategy == WorkflowStrategy.SYSTEMATIC:
                return await self._generate_systematic_phases(workflow, options)
            elif options.strategy == WorkflowStrategy.AGILE:
                return await self._generate_agile_phases(workflow, options)
            elif options.strategy == WorkflowStrategy.MVP:
                return await self._generate_mvp_phases(workflow, options)
            else:
                raise ValueError(f"不支持的工作流策略: {options.strategy}")
                
        except Exception as e:
            self.logger.error(f"生成工作流阶段失败: {str(e)}")
            raise
    
    async def _generate_systematic_phases(self, workflow: Workflow, options: WorkflowOptions) -> List[WorkflowPhase]:
        """生成系统化策略的阶段 - 修复循环依赖问题"""
        try:
            # 1. 使用独立的工作量计算器获取基础工作量
            base_effort = self.effort_calculator.calculate_base_effort(
                workflow.requirements, 
                WorkflowStrategy.SYSTEMATIC
            )
            
            # 2. 使用阶段分配器获取各阶段工作量
            phase_efforts = self.phase_distributor.distribute_effort_to_phases(
                base_effort,
                WorkflowStrategy.SYSTEMATIC
            )
            
            # 3. 生成阶段
            phases = []
            
            phase1 = WorkflowPhase(
                name="需求分析",
                description="深入分析PRD结构和验收标准",
                order=1,
                estimated_effort=phase_efforts.get("需求分析", int(base_effort * 0.15))
            )
            phases.append(phase1)
            
            phase2 = WorkflowPhase(
                name="架构设计",
                description="系统设计和组件架构规划",
                order=2,
                estimated_effort=phase_efforts.get("架构设计", int(base_effort * 0.20))
            )
            phases.append(phase2)
            
            phase3 = WorkflowPhase(
                name="依赖映射",
                description="识别所有内部和外部依赖关系",
                order=3,
                estimated_effort=phase_efforts.get("依赖映射", int(base_effort * 0.10))
            )
            phases.append(phase3)
            
            phase4 = WorkflowPhase(
                name="核心开发",
                description="按序实施各功能模块",
                order=4,
                estimated_effort=phase_efforts.get("核心开发", int(base_effort * 0.35))
            )
            phases.append(phase4)
            
            phase5 = WorkflowPhase(
                name="测试验证",
                description="全面测试和质量保证",
                order=5,
                estimated_effort=phase_efforts.get("测试验证", int(base_effort * 0.15))
            )
            phases.append(phase5)
            
            phase6 = WorkflowPhase(
                name="部署上线",
                description="生产部署和监控策略",
                order=6,
                estimated_effort=phase_efforts.get("部署上线", int(base_effort * 0.05))
            )
            phases.append(phase6)
            
            # 为每个阶段生成里程碑
            for phase in phases:
                phase.milestones = await self._generate_phase_milestones(phase, workflow, options)
            
            self.logger.info(f"系统化策略生成 {len(phases)} 个阶段，总工作量 {base_effort}h")
            return phases
            
        except Exception as e:
            self.logger.error(f"生成系统化阶段失败: {str(e)}")
            raise
    
    async def _generate_agile_phases(self, workflow: Workflow, options: WorkflowOptions) -> List[WorkflowPhase]:
        """生成敏捷策略的阶段 - 修复循环依赖问题"""
        try:
            # 1. 使用独立工作量计算
            base_effort = self.effort_calculator.calculate_base_effort(
                workflow.requirements, 
                WorkflowStrategy.AGILE
            )
            
            # 2. 获取阶段分配
            phase_efforts = self.phase_distributor.distribute_effort_to_phases(
                base_effort,
                WorkflowStrategy.AGILE
            )
            
            phases = []
            
            # 阶段1: Epic分解
            phase1 = WorkflowPhase(
                name="Epic分解",
                description="将PRD转换为用户故事和Epic",
                order=1,
                estimated_effort=phase_efforts.get("Epic分解", int(base_effort * 0.10))
            )
            phases.append(phase1)
            
            # 阶段2: Sprint规划
            phase2 = WorkflowPhase(
                name="Sprint规划",
                description="组织工作为迭代Sprint",
                order=2,
                estimated_effort=phase_efforts.get("Sprint规划", int(base_effort * 0.05))
            )
            phases.append(phase2)
            
            # 阶段3: MVP定义
            phase3 = WorkflowPhase(
                name="MVP定义",
                description="识别最小可行产品范围",
                order=3,
                estimated_effort=phase_efforts.get("MVP定义", int(base_effort * 0.15))
            )
            phases.append(phase3)
            
            # 阶段4: 迭代开发
            phase4 = WorkflowPhase(
                name="迭代开发",
                description="敏捷迭代开发和交付",
                order=4,
                estimated_effort=phase_efforts.get("迭代开发", int(base_effort * 0.50))
            )
            phases.append(phase4)
            
            # 阶段5: 持续集成
            phase5 = WorkflowPhase(
                name="持续集成",
                description="CI/CD流程和自动化测试",
                order=5,
                estimated_effort=phase_efforts.get("持续集成", int(base_effort * 0.15))
            )
            phases.append(phase5)
            
            # 阶段6: 回顾优化
            phase6 = WorkflowPhase(
                name="回顾优化",
                description="项目回顾和持续改进",
                order=6,
                estimated_effort=phase_efforts.get("回顾优化", int(base_effort * 0.05))
            )
            phases.append(phase6)
            
            # 为每个阶段生成里程碑
            for phase in phases:
                phase.milestones = await self._generate_phase_milestones(phase, workflow, options)
            
            self.logger.info(f"敏捷策略生成 {len(phases)} 个阶段，总工作量 {base_effort}h")
            return phases
            
        except Exception as e:
            self.logger.error(f"生成敏捷阶段失败: {str(e)}")
            raise
    
    async def _generate_mvp_phases(self, workflow: Workflow, options: WorkflowOptions) -> List[WorkflowPhase]:
        """生成MVP策略的阶段 - 修复循环依赖问题"""
        try:
            # 1. 使用独立工作量计算
            base_effort = self.effort_calculator.calculate_base_effort(
                workflow.requirements, 
                WorkflowStrategy.MVP
            )
            
            # 2. 获取阶段分配
            phase_efforts = self.phase_distributor.distribute_effort_to_phases(
                base_effort,
                WorkflowStrategy.MVP
            )
            
            phases = []
            
            # 阶段1: 核心功能识别
            phase1 = WorkflowPhase(
                name="核心功能识别",
                description="精简到基本功能",
                order=1,
                estimated_effort=phase_efforts.get("核心功能识别", int(base_effort * 0.20))
            )
            phases.append(phase1)
            
            # 阶段2: 快速原型
            phase2 = WorkflowPhase(
                name="快速原型",
                description="快速验证和反馈",
                order=2,
                estimated_effort=phase_efforts.get("快速原型", int(base_effort * 0.35))
            )
            phases.append(phase2)
            
            # 阶段3: 技术债务规划
            phase3 = WorkflowPhase(
                name="技术债务规划",
                description="识别和规划技术债务",
                order=3,
                estimated_effort=phase_efforts.get("技术债务规划", int(base_effort * 0.10))
            )
            phases.append(phase3)
            
            # 阶段4: 验证指标
            phase4 = WorkflowPhase(
                name="验证指标",
                description="设定和测量成功指标",
                order=4,
                estimated_effort=phase_efforts.get("验证指标", int(base_effort * 0.15))
            )
            phases.append(phase4)
            
            # 阶段5: 扩展路线图
            phase5 = WorkflowPhase(
                name="扩展路线图",
                description="为下一阶段制定计划",
                order=5,
                estimated_effort=phase_efforts.get("扩展路线图", int(base_effort * 0.15))
            )
            phases.append(phase5)
            
            # 阶段6: 反馈集成
            phase6 = WorkflowPhase(
                name="反馈集成",
                description="整合用户反馈到产品中",
                order=6,
                estimated_effort=phase_efforts.get("反馈集成", int(base_effort * 0.05))
            )
            phases.append(phase6)
            
            # 为每个阶段生成里程碑
            for phase in phases:
                phase.milestones = await self._generate_phase_milestones(phase, workflow, options)
            
            self.logger.info(f"MVP策略生成 {len(phases)} 个阶段，总工作量 {base_effort}h")
            return phases
            
        except Exception as e:
            self.logger.error(f"生成MVP阶段失败: {str(e)}")
            raise
    
    async def _generate_phase_milestones(self, phase: WorkflowPhase, workflow: Workflow, options: WorkflowOptions) -> List[WorkflowMilestone]:
        """为阶段生成里程碑"""
        # 这里简化实现，实际应该根据具体阶段内容生成详细的里程碑
        milestones = []
        
        milestone_count = max(2, min(5, phase.estimated_effort // 8))  # 每8小时一个里程碑
        effort_per_milestone = phase.estimated_effort // milestone_count
        
        for i in range(milestone_count):
            milestone = WorkflowMilestone(
                name=f"{phase.name} - 里程碑 {i+1}",
                description=f"{phase.name}阶段的第{i+1}个关键节点",
                estimated_effort=effort_per_milestone,
                priority=Priority.MEDIUM,
                order=i+1
            )
            milestones.append(milestone)
        
        return milestones
    
    async def _analyze_dependencies(self, workflow: Workflow) -> List[Dependency]:
        """分析依赖关系"""
        # 简化实现 - 实际应该分析需求间的依赖关系
        dependencies = []
        
        # 基于需求描述识别外部依赖
        for req in workflow.requirements:
            desc = req.description.lower()
            if any(keyword in desc for keyword in ['第三方', 'api', '集成', '外部']):
                dep = Dependency(
                    name=f"外部依赖 - {req.title}",
                    type="external",
                    description=f"需求 {req.title} 涉及的外部依赖",
                    criticality=req.priority
                )
                dependencies.append(dep)
        
        return dependencies
    
    async def _assess_risks(self, workflow: Workflow) -> List[Risk]:
        """评估风险"""
        risks = []
        
        # 基于复杂度评估技术风险
        if workflow.complexity_analysis:
            if workflow.complexity_analysis.overall_complexity == ComplexityLevel.ENTERPRISE:
                risk = Risk(
                    name="技术复杂度风险",
                    description="项目技术复杂度较高，可能遇到技术挑战",
                    category="technical",
                    likelihood=RiskLevel.HIGH,
                    impact=RiskLevel.HIGH,
                    mitigation_strategies=["增加技术调研时间", "引入专家顾问", "制定技术方案备选"]
                )
                risks.append(risk)
        
        # 基于工作量评估时间风险
        if workflow.effort_estimation and workflow.effort_estimation.total_hours > 200:
            risk = Risk(
                name="项目时间风险",
                description="项目工作量较大，存在延期风险",
                category="timeline",
                likelihood=RiskLevel.MEDIUM,
                impact=RiskLevel.HIGH,
                mitigation_strategies=["分阶段交付", "增加人员", "并行开发"]
            )
            risks.append(risk)
        
        return risks
    
    async def _identify_parallel_streams(self, workflow: Workflow) -> List[ParallelStream]:
        """识别并行工作流"""
        # 简化实现 - 基于人格类型识别可并行的工作
        streams = []
        
        if len(workflow.phases) > 2:
            # 前端和后端可以并行开发
            frontend_stream = ParallelStream(
                name="前端开发流",
                description="UI相关的并行开发流",
                phases=[phase.id for phase in workflow.phases[1:3]]  # 架构设计后可以开始
            )
            streams.append(frontend_stream)
            
            backend_stream = ParallelStream(
                name="后端开发流", 
                description="API和服务相关的并行开发流",
                phases=[phase.id for phase in workflow.phases[1:3]]
            )
            streams.append(backend_stream)
        
        return streams
    
    async def _calculate_critical_path(self, workflow: Workflow) -> CriticalPath:
        """计算关键路径"""
        # 简化实现 - 实际应该基于依赖关系计算
        total_effort = workflow.get_total_effort()
        estimated_duration = timedelta(days=total_effort // 8)  # 假设每天8小时
        
        critical_path = CriticalPath(
            phases=[phase.id for phase in workflow.phases],
            total_duration=estimated_duration,
            total_effort=total_effort,
            bottlenecks=[phase.id for phase in workflow.phases if phase.estimated_effort and phase.estimated_effort > total_effort * 0.3],
            optimization_opportunities=["考虑并行开发", "优化资源分配", "简化复杂需求"]
        )
        
        return critical_path
    
    async def _activate_personas(self, workflow: Workflow, options: WorkflowOptions) -> List[PersonaType]:
        """激活专家人格"""
        activated_personas = []
        
        if options.forced_persona:
            activated_personas.append(options.forced_persona)
            return activated_personas
        
        # 基于需求类别激活相应人格
        if workflow.requirement_categories:
            if workflow.requirement_categories.ui_requirements:
                activated_personas.append(PersonaType.FRONTEND)
            if workflow.requirement_categories.api_requirements:
                activated_personas.append(PersonaType.BACKEND)
            if workflow.requirement_categories.security_requirements:
                activated_personas.append(PersonaType.SECURITY)
            if workflow.requirement_categories.performance_requirements:
                activated_personas.append(PersonaType.PERFORMANCE)
        
        # 基于复杂度激活架构师
        if (workflow.complexity_analysis and 
            workflow.complexity_analysis.overall_complexity in [ComplexityLevel.COMPLEX, ComplexityLevel.ENTERPRISE]):
            activated_personas.append(PersonaType.ARCHITECT)
        
        # 如果没有激活任何人格，默认激活后端
        if not activated_personas:
            activated_personas.append(PersonaType.BACKEND)
        
        return activated_personas
    
    async def _integrate_mcp_services(self, workflow: Workflow, options: WorkflowOptions) -> MCPResults:
        """集成MCP服务"""
        # 这里是占位实现 - 实际需要真正的MCP集成
        results = MCPResults()
        
        if options.enable_context7 or options.enable_all_mcp:
            results.context7_results = {"frameworks": "识别的框架模式"}
        
        if options.enable_sequential or options.enable_all_mcp:
            results.sequential_results = {"analysis": "复杂分析结果"}
            
        if options.enable_magic or options.enable_all_mcp:
            results.magic_results = {"ui_components": "UI组件建议"}
            
        return results
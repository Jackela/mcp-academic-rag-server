"""
工作流输出格式化器

将工作流对象格式化为不同的输出格式（Roadmap、Tasks、Detailed）
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

from .workflow_models import (
    Workflow, WorkflowPhase, WorkflowMilestone, WorkflowStep,
    OutputFormat, PersonaType, Priority, ComplexityLevel, Risk, Dependency
)


class WorkflowFormatter:
    """工作流格式化器"""
    
    def __init__(self):
        self.logger = logging.getLogger("WorkflowFormatter")
    
    def format_workflow(self, workflow: Workflow, output_format: OutputFormat) -> str:
        """格式化工作流为指定格式"""
        try:
            if output_format == OutputFormat.ROADMAP:
                return self._format_roadmap(workflow)
            elif output_format == OutputFormat.TASKS:
                return self._format_tasks(workflow)
            elif output_format == OutputFormat.DETAILED:
                return self._format_detailed(workflow)
            else:
                raise ValueError(f"不支持的输出格式: {output_format}")
                
        except Exception as e:
            self.logger.error(f"格式化工作流失败: {str(e)}")
            raise
    
    def _format_roadmap(self, workflow: Workflow) -> str:
        """格式化为路线图格式"""
        try:
            output = []
            
            # 标题和概述
            output.append(f"# {workflow.name} - 实施路线图")
            output.append("")
            output.append(f"**生成时间**: {workflow.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
            output.append(f"**策略**: {workflow.strategy.value.title()}")
            output.append(f"**预估总工作量**: {workflow.get_total_effort()} 小时")
            
            if workflow.complexity_analysis:
                output.append(f"**复杂度级别**: {workflow.complexity_analysis.overall_complexity.value.title()}")
            
            if workflow.activated_personas:
                personas_str = ", ".join([p.value.title() for p in workflow.activated_personas])
                output.append(f"**涉及专家**: {personas_str}")
            
            output.append("")
            
            if workflow.description:
                output.append("## 📋 项目概述")
                output.append(workflow.description)
                output.append("")
            
            # 关键指标
            if workflow.complexity_analysis or workflow.effort_estimation:
                output.append("## 📊 关键指标")
                
                if workflow.effort_estimation:
                    output.append(f"- **总工作量**: {workflow.effort_estimation.total_hours} 小时")
                    output.append(f"- **置信度**: {workflow.effort_estimation.confidence_level * 100:.0f}%")
                    
                    if workflow.effort_estimation.breakdown_by_persona:
                        output.append("- **人员分工**:")
                        for persona, hours in workflow.effort_estimation.breakdown_by_persona.items():
                            output.append(f"  - {persona.value.title()}: {hours} 小时")
                
                if workflow.complexity_analysis:
                    output.append(f"- **技术复杂度**: {workflow.complexity_analysis.technical_complexity * 100:.0f}%")
                    output.append(f"- **集成复杂度**: {workflow.complexity_analysis.integration_complexity * 100:.0f}%")
                
                output.append("")
            
            # 阶段路线图
            output.append("## 🚀 实施阶段")
            output.append("")
            
            total_weeks = max(1, workflow.get_total_effort() // 40)  # 假设每周40小时
            current_week = 1
            
            for phase in workflow.phases:
                phase_weeks = max(1, (phase.estimated_effort or 0) // 40)
                week_range = f"第 {current_week} 周" if phase_weeks == 1 else f"第 {current_week}-{current_week + phase_weeks - 1} 周"
                
                output.append(f"### {phase.name} ({week_range})")
                output.append("")
                output.append(f"**预估工作量**: {phase.estimated_effort or 0} 小时")
                
                if phase.personas_involved:
                    personas_str = ", ".join([p.value.title() for p in phase.personas_involved])
                    output.append(f"**涉及角色**: {personas_str}")
                
                if phase.dependencies:
                    output.append(f"**依赖项**: {len(phase.dependencies)} 个")
                
                if phase.risks:
                    high_risks = [r for r in workflow.risks if r.id in phase.risks and r.likelihood.value == 'high']
                    if high_risks:
                        output.append(f"**高风险项**: {len(high_risks)} 个")
                
                output.append("")
                
                if phase.description:
                    output.append(phase.description)
                    output.append("")
                
                # 里程碑
                if phase.milestones:
                    output.append("#### 🎯 关键里程碑")
                    for milestone in phase.milestones:
                        priority_emoji = self._get_priority_emoji(milestone.priority)
                        output.append(f"- [ ] {priority_emoji} {milestone.name} ({milestone.estimated_effort or 0}h)")
                        if milestone.description:
                            output.append(f"  - {milestone.description}")
                    output.append("")
                
                # 成功标准
                if phase.milestones and any(m.success_criteria for m in phase.milestones):
                    output.append("#### ✅ 成功标准")
                    for milestone in phase.milestones:
                        if milestone.success_criteria:
                            for criteria in milestone.success_criteria:
                                output.append(f"- {criteria}")
                    output.append("")
                
                current_week += phase_weeks
            
            # 风险和缓解措施
            if workflow.risks:
                output.append("## ⚠️ 风险分析")
                output.append("")
                
                high_risks = [r for r in workflow.risks if r.likelihood.value == 'high' or r.impact.value == 'high']
                medium_risks = [r for r in workflow.risks if r not in high_risks and (r.likelihood.value == 'medium' or r.impact.value == 'medium')]
                
                if high_risks:
                    output.append("### 🔴 高风险项")
                    for risk in high_risks:
                        output.append(f"- **{risk.name}**: {risk.description}")
                        if risk.mitigation_strategies:
                            output.append(f"  - 缓解策略: {'; '.join(risk.mitigation_strategies)}")
                    output.append("")
                
                if medium_risks:
                    output.append("### 🟡 中等风险项")
                    for risk in medium_risks:
                        output.append(f"- **{risk.name}**: {risk.description}")
                    output.append("")
            
            # 并行工作流
            if workflow.parallel_streams:
                output.append("## 🔄 并行工作流")
                output.append("")
                output.append("以下工作可以并行进行，以加速项目进度：")
                output.append("")
                
                for stream in workflow.parallel_streams:
                    output.append(f"### {stream.name}")
                    output.append(f"- **描述**: {stream.description}")
                    if stream.estimated_effort:
                        output.append(f"- **预估工作量**: {stream.estimated_effort} 小时")
                    if stream.required_team_size > 1:
                        output.append(f"- **需要团队规模**: {stream.required_team_size} 人")
                    output.append("")
            
            # 关键路径
            if workflow.critical_path:
                output.append("## 🎯 关键路径")
                output.append("")
                if workflow.critical_path.total_duration:
                    days = workflow.critical_path.total_duration.days
                    output.append(f"**关键路径总时长**: {days} 天")
                if workflow.critical_path.total_effort:
                    output.append(f"**关键路径总工作量**: {workflow.critical_path.total_effort} 小时")
                
                if workflow.critical_path.bottlenecks:
                    output.append("")
                    output.append("### 🚧 潜在瓶颈")
                    for bottleneck_id in workflow.critical_path.bottlenecks:
                        phase = next((p for p in workflow.phases if p.id == bottleneck_id), None)
                        if phase:
                            output.append(f"- {phase.name}: 工作量较大，可能影响整体进度")
                    output.append("")
                
                if workflow.critical_path.optimization_opportunities:
                    output.append("### 💡 优化建议")
                    for opportunity in workflow.critical_path.optimization_opportunities:
                        output.append(f"- {opportunity}")
                    output.append("")
            
            return "\n".join(output)
            
        except Exception as e:
            self.logger.error(f"格式化路线图失败: {str(e)}")
            raise
    
    def _format_tasks(self, workflow: Workflow) -> str:
        """格式化为任务格式"""
        try:
            output = []
            
            # 标题和概述
            output.append(f"# {workflow.name} - 实施任务清单")
            output.append("")
            output.append(f"**生成时间**: {workflow.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
            output.append(f"**策略**: {workflow.strategy.value.title()}")
            output.append(f"**总任务数**: {sum(len(phase.milestones) for phase in workflow.phases)}")
            output.append(f"**预估总工作量**: {workflow.get_total_effort()} 小时")
            output.append("")
            
            # 任务概览
            output.append("## 📊 任务概览")
            output.append("")
            
            # 按优先级统计
            all_milestones = [m for phase in workflow.phases for m in phase.milestones]
            priority_stats = {}
            for milestone in all_milestones:
                priority = milestone.priority.value
                priority_stats[priority] = priority_stats.get(priority, 0) + 1
            
            for priority in ['high', 'medium', 'low']:
                count = priority_stats.get(priority, 0)
                emoji = self._get_priority_emoji(Priority(priority))
                output.append(f"- {emoji} **{priority.title()}优先级**: {count} 个任务")
            
            output.append("")
            
            # 按阶段列出任务
            output.append("## 📋 分阶段任务")
            output.append("")
            
            for phase_idx, phase in enumerate(workflow.phases, 1):
                output.append(f"### Epic {phase_idx}: {phase.name}")
                output.append("")
                output.append(f"**描述**: {phase.description or '待补充描述'}")
                output.append(f"**预估工作量**: {phase.estimated_effort or 0} 小时")
                
                if phase.personas_involved:
                    personas_str = ", ".join([p.value.title() for p in phase.personas_involved])
                    output.append(f"**涉及角色**: {personas_str}")
                
                output.append("")
                
                # 阶段任务
                for milestone_idx, milestone in enumerate(phase.milestones, 1):
                    priority_emoji = self._get_priority_emoji(milestone.priority)
                    complexity_indicator = self._get_complexity_indicator(milestone.estimated_effort or 0)
                    
                    output.append(f"#### Story {phase_idx}.{milestone_idx}: {milestone.name}")
                    output.append(f"**优先级**: {priority_emoji} {milestone.priority.value.title()} | "
                                f"**工作量**: {milestone.estimated_effort or 0}h {complexity_indicator}")
                    
                    if milestone.dependencies:
                        output.append(f"**依赖**: {len(milestone.dependencies)} 个前置条件")
                    
                    output.append("")
                    
                    if milestone.description:
                        output.append(milestone.description)
                        output.append("")
                    
                    # 任务步骤
                    if milestone.steps:
                        for step_idx, step in enumerate(milestone.steps, 1):
                            persona_tag = f"({step.persona.value})" if step.persona else ""
                            output.append(f"- [ ] **步骤 {step_idx}**: {step.name} {persona_tag}")
                            if step.estimated_effort:
                                output.append(f"  - 预估: {step.estimated_effort}h")
                            if step.description:
                                output.append(f"  - 描述: {step.description}")
                    else:
                        # 如果没有详细步骤，生成基础任务
                        output.append(f"- [ ] 设计和规划 ({milestone.estimated_effort // 4 or 1}h)")
                        output.append(f"- [ ] 开发实现 ({milestone.estimated_effort // 2 or 2}h)")  
                        output.append(f"- [ ] 测试验证 ({milestone.estimated_effort // 4 or 1}h)")
                    
                    # 验收标准
                    if milestone.success_criteria:
                        output.append("")
                        output.append("**验收标准**:")
                        for criteria in milestone.success_criteria:
                            output.append(f"- [ ] {criteria}")
                    
                    output.append("")
            
            # 依赖关系图
            if workflow.dependencies:
                output.append("## 🔗 依赖关系")
                output.append("")
                
                external_deps = [d for d in workflow.dependencies if d.type == 'external']
                internal_deps = [d for d in workflow.dependencies if d.type == 'internal']
                
                if external_deps:
                    output.append("### 外部依赖")
                    for dep in external_deps:
                        criticality_emoji = self._get_priority_emoji(dep.criticality)
                        output.append(f"- {criticality_emoji} **{dep.name}**: {dep.description}")
                        if dep.owner:
                            output.append(f"  - 负责人: {dep.owner}")
                        if dep.estimated_resolution_time:
                            output.append(f"  - 预估解决时间: {dep.estimated_resolution_time}h")
                    output.append("")
                
                if internal_deps:
                    output.append("### 内部依赖")
                    for dep in internal_deps:
                        criticality_emoji = self._get_priority_emoji(dep.criticality)
                        output.append(f"- {criticality_emoji} **{dep.name}**: {dep.description}")
                    output.append("")
            
            return "\n".join(output)
            
        except Exception as e:
            self.logger.error(f"格式化任务清单失败: {str(e)}")
            raise
    
    def _format_detailed(self, workflow: Workflow) -> str:
        """格式化为详细格式"""
        try:
            output = []
            
            # 标题和详细概述
            output.append(f"# {workflow.name} - 详细实施工作流")
            output.append("")
            
            # 元数据部分
            output.append("## 📋 项目元数据")
            output.append("")
            output.append(f"- **项目名称**: {workflow.name}")
            output.append(f"- **生成时间**: {workflow.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
            output.append(f"- **工作流策略**: {workflow.strategy.value.title()}")
            output.append(f"- **版本**: {workflow.version}")
            if workflow.prd_structure:
                output.append(f"- **PRD标题**: {workflow.prd_structure.title}")
                if workflow.prd_structure.author:
                    output.append(f"- **PRD作者**: {workflow.prd_structure.author}")
            output.append("")
            
            # 项目概述
            if workflow.description or (workflow.prd_structure and workflow.prd_structure.overview):
                output.append("## 📖 项目概述")
                output.append("")
                if workflow.description:
                    output.append(workflow.description)
                elif workflow.prd_structure and workflow.prd_structure.overview:
                    output.append(workflow.prd_structure.overview)
                output.append("")
            
            # 需求分析
            if workflow.requirements:
                output.append("## 📝 需求分析")
                output.append("")
                output.append(f"**需求总数**: {len(workflow.requirements)}")
                
                if workflow.requirement_categories:
                    output.append("")
                    output.append("### 需求分类")
                    cats = workflow.requirement_categories
                    output.append(f"- **功能性需求**: {len(cats.functional_requirements)}")
                    output.append(f"- **非功能性需求**: {len(cats.non_functional_requirements)}")
                    output.append(f"- **UI需求**: {len(cats.ui_requirements)}")
                    output.append(f"- **API需求**: {len(cats.api_requirements)}")
                    output.append(f"- **数据需求**: {len(cats.data_requirements)}")
                    output.append(f"- **安全需求**: {len(cats.security_requirements)}")
                    output.append(f"- **性能需求**: {len(cats.performance_requirements)}")
                
                output.append("")
                output.append("### 核心需求详情")
                for i, req in enumerate(workflow.requirements[:5], 1):  # 只显示前5个需求
                    priority_emoji = self._get_priority_emoji(req.priority)
                    complexity_badge = req.complexity.value.title()
                    
                    output.append(f"#### {i}. {req.title}")
                    output.append(f"**优先级**: {priority_emoji} {req.priority.value.title()} | "
                              f"**复杂度**: {complexity_badge}")
                    output.append("")
                    output.append(req.description)
                    
                    if req.acceptance_criteria:
                        output.append("")
                        output.append("**验收标准**:")
                        for criteria in req.acceptance_criteria:
                            output.append(f"- {criteria}")
                    
                    output.append("")
                
                if len(workflow.requirements) > 5:
                    output.append(f"*... 还有 {len(workflow.requirements) - 5} 个需求未显示*")
                    output.append("")
            
            # 复杂度和工作量分析
            if workflow.complexity_analysis or workflow.effort_estimation:
                output.append("## 📊 技术分析")
                output.append("")
                
                if workflow.complexity_analysis:
                    ca = workflow.complexity_analysis
                    output.append("### 复杂度分析")
                    output.append(f"- **整体复杂度**: {ca.overall_complexity.value.title()}")
                    output.append(f"- **技术复杂度**: {ca.technical_complexity * 100:.1f}%")
                    output.append(f"- **集成复杂度**: {ca.integration_complexity * 100:.1f}%")
                    output.append(f"- **UI复杂度**: {ca.ui_complexity * 100:.1f}%")
                    output.append(f"- **业务逻辑复杂度**: {ca.business_logic_complexity * 100:.1f}%")
                    output.append(f"- **数据复杂度**: {ca.data_complexity * 100:.1f}%")
                    
                    if ca.complexity_factors:
                        output.append("")
                        output.append("**复杂度因子**:")
                        for factor in ca.complexity_factors:
                            output.append(f"- {factor}")
                    
                    if ca.simplification_opportunities:
                        output.append("")
                        output.append("**简化建议**:")
                        for opportunity in ca.simplification_opportunities:
                            output.append(f"- {opportunity}")
                    
                    output.append("")
                
                if workflow.effort_estimation:
                    ee = workflow.effort_estimation
                    output.append("### 工作量估算")
                    output.append(f"- **总工作量**: {ee.total_hours} 小时")
                    output.append(f"- **置信度**: {ee.confidence_level * 100:.0f}%")
                    output.append(f"- **估算方法**: {ee.estimation_method}")
                    output.append(f"- **缓冲比例**: {ee.buffer_percentage * 100:.0f}%")
                    
                    if ee.breakdown_by_persona:
                        output.append("")
                        output.append("**人员分工明细**:")
                        for persona, hours in ee.breakdown_by_persona.items():
                            percentage = (hours / ee.total_hours) * 100
                            output.append(f"- **{persona.value.title()}**: {hours}h ({percentage:.1f}%)")
                    
                    output.append("")
            
            # 专家人格激活
            if workflow.activated_personas:
                output.append("## 👥 专家团队配置")
                output.append("")
                output.append("### 激活的专家人格")
                
                persona_descriptions = {
                    PersonaType.FRONTEND: "负责UI/UX设计、用户体验优化、前端性能调优",
                    PersonaType.BACKEND: "负责API设计、数据库架构、服务端性能优化",
                    PersonaType.ARCHITECT: "负责系统架构设计、技术选型、扩展性规划",
                    PersonaType.SECURITY: "负责安全架构设计、漏洞评估、合规性检查",
                    PersonaType.DEVOPS: "负责CI/CD流程、基础设施管理、监控告警",
                    PersonaType.QA: "负责测试策略制定、质量保证、自动化测试",
                    PersonaType.SCRIBE: "负责文档编写、知识管理、培训材料制作",
                    PersonaType.PERFORMANCE: "负责性能优化、瓶颈分析、监控指标设计"
                }
                
                for persona in workflow.activated_personas:
                    description = persona_descriptions.get(persona, "专业领域专家")
                    output.append(f"- **{persona.value.title()}**: {description}")
                
                if workflow.persona_recommendations:
                    output.append("")
                    output.append("### 专家建议")
                    for persona, recommendation in workflow.persona_recommendations.items():
                        output.append(f"- **{persona.value.title()}**: {recommendation}")
                
                output.append("")
            
            # MCP集成结果
            if workflow.mcp_results:
                output.append("## 🔌 MCP服务集成结果")
                output.append("")
                
                mr = workflow.mcp_results
                if mr.context7_results:
                    output.append("### Context7 框架模式分析")
                    for key, value in mr.context7_results.items():
                        output.append(f"- **{key}**: {value}")
                    output.append("")
                
                if mr.sequential_results:
                    output.append("### Sequential 复杂分析结果")
                    for key, value in mr.sequential_results.items():
                        output.append(f"- **{key}**: {value}")
                    output.append("")
                
                if mr.magic_results:
                    output.append("### Magic UI组件建议")
                    for key, value in mr.magic_results.items():
                        output.append(f"- **{key}**: {value}")
                    output.append("")
                
                if mr.integration_recommendations:
                    output.append("### 集成建议")
                    for recommendation in mr.integration_recommendations:
                        output.append(f"- {recommendation}")
                    output.append("")
            
            # 详细实施步骤
            output.append("## 🚀 详细实施步骤")
            output.append("")
            
            for phase_idx, phase in enumerate(workflow.phases, 1):
                output.append(f"### 阶段 {phase_idx}: {phase.name}")
                output.append("")
                
                # 阶段概述
                output.append("#### 📋 阶段概述")
                output.append(f"- **描述**: {phase.description or '待补充描述'}")
                output.append(f"- **预估工作量**: {phase.estimated_effort or 0} 小时")
                
                if phase.estimated_effort:
                    estimated_days = (phase.estimated_effort + 7) // 8  # 向上取整到天
                    output.append(f"- **预估工期**: {estimated_days} 工作日")
                
                if phase.personas_involved:
                    personas_str = ", ".join([p.value.title() for p in phase.personas_involved])
                    output.append(f"- **涉及角色**: {personas_str}")
                
                if phase.dependencies:
                    output.append(f"- **前置依赖**: {len(phase.dependencies)} 个")
                
                if phase.risks:
                    phase_risks = [r for r in workflow.risks if r.id in phase.risks]
                    high_risks = [r for r in phase_risks if r.likelihood.value == 'high' or r.impact.value == 'high']
                    if high_risks:
                        output.append(f"- **高风险项**: {len(high_risks)} 个")
                
                output.append("")
                
                # 详细里程碑
                if phase.milestones:
                    output.append("#### 🎯 详细里程碑")
                    output.append("")
                    
                    for milestone_idx, milestone in enumerate(phase.milestones, 1):
                        output.append(f"##### 里程碑 {phase_idx}.{milestone_idx}: {milestone.name}")
                        
                        priority_emoji = self._get_priority_emoji(milestone.priority)
                        complexity_indicator = self._get_complexity_indicator(milestone.estimated_effort or 0)
                        
                        output.append(f"**优先级**: {priority_emoji} {milestone.priority.value.title()}")
                        output.append(f"**预估工作量**: {milestone.estimated_effort or 0} 小时 {complexity_indicator}")
                        
                        if milestone.personas_involved:
                            personas_str = ", ".join([p.value.title() for p in milestone.personas_involved])
                            output.append(f"**专家角色**: {personas_str}")
                        
                        if milestone.dependencies:
                            output.append(f"**依赖项**: {len(milestone.dependencies)} 个")
                        
                        output.append("")
                        
                        if milestone.description:
                            output.append(f"**描述**: {milestone.description}")
                            output.append("")
                        
                        # 实施步骤
                        if milestone.steps:
                            output.append("**实施步骤**:")
                            output.append("")
                            
                            for step_idx, step in enumerate(milestone.steps, 1):
                                output.append(f"**步骤 {step_idx}: {step.name}**")
                                
                                if step.persona:
                                    output.append(f"*负责人*: {step.persona.value.title()}")
                                
                                if step.estimated_effort:
                                    output.append(f"*预估时间*: {step.estimated_effort} 小时")
                                
                                if step.complexity:
                                    output.append(f"*复杂度*: {step.complexity.value.title()}")
                                
                                output.append("")
                                
                                if step.description:
                                    output.append(step.description)
                                    output.append("")
                                
                                # 工具和示例
                                if step.tools_required:
                                    output.append("*所需工具*:")
                                    for tool in step.tools_required:
                                        output.append(f"- {tool}")
                                    output.append("")
                                
                                if step.code_examples:
                                    output.append("*代码示例*:")
                                    for example in step.code_examples:
                                        output.append(f"```\n{example}\n```")
                                    output.append("")
                                
                                if step.mcp_context:
                                    output.append("*MCP上下文*:")
                                    for server, context in step.mcp_context.items():
                                        output.append(f"- **{server}**: {context}")
                                    output.append("")
                                
                                # 交付物
                                if step.deliverables:
                                    output.append("*交付物*:")
                                    for deliverable in step.deliverables:
                                        output.append(f"- [ ] {deliverable}")
                                    output.append("")
                                
                                # 验收标准
                                if step.acceptance_criteria:
                                    output.append("*验收标准*:")
                                    for criteria in step.acceptance_criteria:
                                        output.append(f"- [ ] {criteria}")
                                    output.append("")
                        else:
                            # 生成默认步骤
                            output.append("**实施步骤**:")
                            output.append("")
                            output.append("1. **需求分析和设计** (25%)")
                            output.append("   - 详细分析功能需求")
                            output.append("   - 制定技术方案")
                            output.append("   - 设计接口和数据结构")
                            output.append("")
                            output.append("2. **核心开发** (50%)")
                            output.append("   - 实现核心功能逻辑")
                            output.append("   - 编写单元测试")
                            output.append("   - 代码审查和优化")
                            output.append("")
                            output.append("3. **集成测试** (15%)")
                            output.append("   - 集成测试和联调")
                            output.append("   - 性能测试和优化")
                            output.append("   - 缺陷修复")
                            output.append("")
                            output.append("4. **部署验收** (10%)")
                            output.append("   - 部署到测试环境")
                            output.append("   - 用户验收测试")
                            output.append("   - 文档更新")
                            output.append("")
                        
                        # 成功标准
                        if milestone.success_criteria:
                            output.append("**成功标准**:")
                            for criteria in milestone.success_criteria:
                                output.append(f"- [ ] {criteria}")
                            output.append("")
                        
                        # 风险评估
                        if milestone.risks:
                            milestone_risks = [r for r in workflow.risks if r.id in milestone.risks]
                            if milestone_risks:
                                output.append("**风险评估**:")
                                for risk in milestone_risks:
                                    risk_level = f"{risk.likelihood.value.title()}/{risk.impact.value.title()}"
                                    output.append(f"- **{risk.name}** (概率/影响: {risk_level})")
                                    output.append(f"  - {risk.description}")
                                    if risk.mitigation_strategies:
                                        output.append(f"  - 缓解措施: {'; '.join(risk.mitigation_strategies)}")
                                output.append("")
                
                output.append("---")
                output.append("")
            
            # 质量保证
            if workflow.quality_gates or workflow.success_metrics:
                output.append("## ✅ 质量保证")
                output.append("")
                
                if workflow.quality_gates:
                    output.append("### 质量门控")
                    for gate in workflow.quality_gates:
                        output.append(f"- [ ] {gate}")
                    output.append("")
                
                if workflow.success_metrics:
                    output.append("### 成功指标")
                    for metric in workflow.success_metrics:
                        output.append(f"- {metric}")
                    output.append("")
                
                if workflow.validation_criteria:
                    output.append("### 验证标准")
                    for criteria in workflow.validation_criteria:
                        output.append(f"- [ ] {criteria}")
                    output.append("")
            
            return "\n".join(output)
            
        except Exception as e:
            self.logger.error(f"格式化详细工作流失败: {str(e)}")
            raise
    
    def _get_priority_emoji(self, priority: Priority) -> str:
        """获取优先级对应的emoji"""
        priority_emojis = {
            Priority.LOW: "🟢",
            Priority.MEDIUM: "🟡", 
            Priority.HIGH: "🔴",
            Priority.CRITICAL: "🚨"
        }
        return priority_emojis.get(priority, "⚪")
    
    def _get_complexity_indicator(self, hours: int) -> str:
        """基于工作量获取复杂度指示符"""
        if hours <= 4:
            return "🟢"  # 简单
        elif hours <= 16:
            return "🟡"  # 中等
        elif hours <= 40:
            return "🔴"  # 复杂
        else:
            return "🚨"  # 非常复杂
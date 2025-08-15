"""
工作量计算器 - 解决循环依赖问题

提供独立的工作量计算功能，避免在工作流生成过程中产生循环依赖。
"""

import logging
from typing import List, Dict, Optional
from .workflow_models import (
    Requirement, ComplexityLevel, PersonaType, WorkflowStrategy,
    EffortEstimation, RequirementCategories
)


class EffortCalculator:
    """独立的工作量计算器"""
    
    def __init__(self):
        self.logger = logging.getLogger("EffortCalculator")
        
        # 基于复杂度的工作量映射表
        self.complexity_effort_map = {
            ComplexityLevel.SIMPLE: 8,      # 8小时
            ComplexityLevel.MEDIUM: 16,     # 16小时
            ComplexityLevel.COMPLEX: 32,    # 32小时
            ComplexityLevel.ENTERPRISE: 64  # 64小时
        }
        
        # 不同策略的系统开销比例
        self.strategy_overhead_map = {
            WorkflowStrategy.SYSTEMATIC: 0.35,  # 系统化策略开销35%
            WorkflowStrategy.AGILE: 0.25,       # 敏捷策略开销25%
            WorkflowStrategy.MVP: 0.15          # MVP策略开销15%
        }
        
        # 专家人格工作量分配比例
        self.persona_allocation_map = {
            PersonaType.BACKEND: 0.35,      # 后端35%
            PersonaType.FRONTEND: 0.25,     # 前端25%
            PersonaType.ARCHITECT: 0.15,    # 架构15%
            PersonaType.SECURITY: 0.10,     # 安全10%
            PersonaType.QA: 0.10,           # 测试10%
            PersonaType.DEVOPS: 0.05        # 运维5%
        }
    
    def calculate_base_effort(
        self, 
        requirements: List[Requirement], 
        strategy: WorkflowStrategy = WorkflowStrategy.SYSTEMATIC
    ) -> int:
        """
        基于需求直接计算基础工作量
        
        Args:
            requirements: 需求列表
            strategy: 工作流策略
            
        Returns:
            基础工作量（小时）
        """
        try:
            # 1. 计算需求核心工作量
            core_effort = 0
            for req in requirements:
                if req.estimated_effort:
                    core_effort += req.estimated_effort
                else:
                    # 基于复杂度估算
                    req_effort = self.complexity_effort_map.get(req.complexity, 16)
                    core_effort += req_effort
            
            # 2. 添加策略相关的系统开销
            overhead_ratio = self.strategy_overhead_map.get(strategy, 0.3)
            system_overhead = int(core_effort * overhead_ratio)
            
            total_effort = core_effort + system_overhead
            
            self.logger.info(f"计算基础工作量: 核心={core_effort}h, 开销={system_overhead}h, 总计={total_effort}h")
            return total_effort
            
        except Exception as e:
            self.logger.error(f"计算基础工作量失败: {str(e)}")
            # 返回默认值
            return len(requirements) * 16 + 40  # 默认每需求16h + 40h开销
    
    def calculate_detailed_estimation(
        self,
        requirements: List[Requirement],
        requirement_categories: Optional[RequirementCategories] = None,
        strategy: WorkflowStrategy = WorkflowStrategy.SYSTEMATIC,
        team_size: int = 1
    ) -> EffortEstimation:
        """
        计算详细的工作量估算
        
        Args:
            requirements: 需求列表
            requirement_categories: 需求分类
            strategy: 工作流策略
            team_size: 团队规模
            
        Returns:
            详细的工作量估算
        """
        try:
            # 1. 计算基础工作量
            base_effort = self.calculate_base_effort(requirements, strategy)
            
            # 2. 团队规模调整
            if team_size > 1:
                # 大团队有沟通开销，但可以并行工作
                communication_overhead = 1 + (team_size - 1) * 0.1  # 每增加1人增加10%开销
                parallelization_benefit = 1 - (team_size - 1) * 0.15  # 每增加1人减少15%时间
                team_adjustment = communication_overhead * parallelization_benefit
                base_effort = int(base_effort * team_adjustment)
            
            # 3. 计算专家人格分工
            persona_breakdown = self._calculate_persona_breakdown(
                requirements, base_effort, requirement_categories
            )
            
            # 4. 计算置信度
            confidence = self._calculate_confidence_level(requirements, strategy)
            
            # 5. 计算缓冲
            buffer_percentage = self._calculate_buffer_percentage(strategy, confidence)
            
            estimation = EffortEstimation(
                total_hours=base_effort,
                breakdown_by_persona=persona_breakdown,
                confidence_level=confidence,
                estimation_method="requirement_based_calculation",
                buffer_percentage=buffer_percentage,
                metadata={
                    'base_effort': base_effort,
                    'team_size': team_size,
                    'strategy': strategy.value,
                    'requirements_count': len(requirements)
                }
            )
            
            self.logger.info(f"详细工作量估算完成: {base_effort}h, 置信度: {confidence:.1%}")
            return estimation
            
        except Exception as e:
            self.logger.error(f"计算详细工作量估算失败: {str(e)}")
            # 返回默认估算
            return EffortEstimation(
                total_hours=len(requirements) * 20,
                confidence_level=0.5,
                estimation_method="fallback_calculation"
            )
    
    def _calculate_persona_breakdown(
        self,
        requirements: List[Requirement],
        total_effort: int,
        requirement_categories: Optional[RequirementCategories] = None
    ) -> Dict[PersonaType, int]:
        """计算专家人格工作量分配"""
        breakdown = {}
        
        try:
            if requirement_categories:
                # 基于需求分类智能分配
                total_reqs = len(requirements)
                if total_reqs == 0:
                    return breakdown
                
                # 后端工作量
                backend_ratio = (len(requirement_categories.api_requirements) + 
                               len(requirement_categories.data_requirements)) / total_reqs
                backend_ratio = max(0.2, min(0.5, backend_ratio))  # 限制在20%-50%
                
                # 前端工作量  
                frontend_ratio = len(requirement_categories.ui_requirements) / total_reqs
                frontend_ratio = max(0.15, min(0.4, frontend_ratio))  # 限制在15%-40%
                
                # 安全工作量
                security_ratio = len(requirement_categories.security_requirements) / total_reqs
                security_ratio = max(0.05, min(0.2, security_ratio))  # 限制在5%-20%
                
                # 其他角色按剩余比例分配
                remaining_ratio = 1.0 - backend_ratio - frontend_ratio - security_ratio
                
                breakdown[PersonaType.BACKEND] = int(total_effort * backend_ratio)
                breakdown[PersonaType.FRONTEND] = int(total_effort * frontend_ratio)
                breakdown[PersonaType.SECURITY] = int(total_effort * security_ratio)
                breakdown[PersonaType.ARCHITECT] = int(total_effort * remaining_ratio * 0.4)
                breakdown[PersonaType.QA] = int(total_effort * remaining_ratio * 0.35)
                breakdown[PersonaType.DEVOPS] = int(total_effort * remaining_ratio * 0.25)
                
            else:
                # 使用默认分配比例
                for persona, ratio in self.persona_allocation_map.items():
                    breakdown[persona] = int(total_effort * ratio)
            
            # 确保总和等于总工作量
            current_total = sum(breakdown.values())
            if current_total != total_effort:
                # 调整最大的分配项
                max_persona = max(breakdown.keys(), key=lambda x: breakdown[x])
                breakdown[max_persona] += (total_effort - current_total)
            
            return breakdown
            
        except Exception as e:
            self.logger.error(f"计算专家人格分配失败: {str(e)}")
            # 返回默认分配
            return {PersonaType.BACKEND: int(total_effort * 0.5)}
    
    def _calculate_confidence_level(
        self,
        requirements: List[Requirement],
        strategy: WorkflowStrategy
    ) -> float:
        """计算估算置信度"""
        confidence = 0.8  # 基础置信度80%
        
        try:
            # 基于需求清晰度调整
            clear_requirements = sum(1 for req in requirements 
                                   if req.description and len(req.description) > 20)
            clarity_ratio = clear_requirements / len(requirements) if requirements else 0
            confidence *= (0.7 + clarity_ratio * 0.3)  # 70%-100%基于清晰度
            
            # 基于策略调整
            strategy_confidence_map = {
                WorkflowStrategy.SYSTEMATIC: 1.0,    # 系统化策略置信度最高
                WorkflowStrategy.AGILE: 0.9,         # 敏捷策略次之
                WorkflowStrategy.MVP: 0.8            # MVP策略最低
            }
            confidence *= strategy_confidence_map.get(strategy, 0.8)
            
            # 基于需求数量调整（需求太少或太多都降低置信度）
            req_count = len(requirements)
            if req_count < 3:
                confidence *= 0.8  # 需求太少
            elif req_count > 20:
                confidence *= 0.9  # 需求太多
            
            return max(0.3, min(0.95, confidence))  # 限制在30%-95%
            
        except Exception as e:
            self.logger.error(f"计算置信度失败: {str(e)}")
            return 0.7  # 默认70%置信度
    
    def _calculate_buffer_percentage(
        self,
        strategy: WorkflowStrategy,
        confidence: float
    ) -> float:
        """计算缓冲比例"""
        try:
            # 基础缓冲比例
            base_buffer = {
                WorkflowStrategy.SYSTEMATIC: 0.15,  # 系统化15%缓冲
                WorkflowStrategy.AGILE: 0.20,       # 敏捷20%缓冲  
                WorkflowStrategy.MVP: 0.25          # MVP 25%缓冲
            }
            
            buffer = base_buffer.get(strategy, 0.2)
            
            # 基于置信度调整缓冲
            if confidence < 0.6:
                buffer += 0.1  # 低置信度增加10%缓冲
            elif confidence > 0.9:
                buffer -= 0.05  # 高置信度减少5%缓冲
            
            return max(0.1, min(0.4, buffer))  # 限制在10%-40%
            
        except Exception as e:
            self.logger.error(f"计算缓冲比例失败: {str(e)}")
            return 0.2  # 默认20%缓冲


class PhaseEffortDistributor:
    """阶段工作量分配器"""
    
    def __init__(self):
        self.logger = logging.getLogger("PhaseEffortDistributor")
        
        # 不同策略的阶段分配模板
        self.phase_distribution_templates = {
            WorkflowStrategy.SYSTEMATIC: {
                "需求分析": 0.15,
                "架构设计": 0.20,
                "依赖映射": 0.10,
                "核心开发": 0.35,
                "测试验证": 0.15,
                "部署上线": 0.05
            },
            WorkflowStrategy.AGILE: {
                "Epic分解": 0.10,
                "Sprint规划": 0.05,
                "MVP定义": 0.15,
                "迭代开发": 0.50,
                "持续集成": 0.15,
                "回顾优化": 0.05
            },
            WorkflowStrategy.MVP: {
                "核心功能识别": 0.20,
                "快速原型": 0.35,
                "技术债务规划": 0.10,
                "验证指标": 0.15,
                "扩展路线图": 0.15,
                "反馈集成": 0.05
            }
        }
    
    def distribute_effort_to_phases(
        self,
        total_effort: int,
        strategy: WorkflowStrategy
    ) -> Dict[str, int]:
        """将总工作量分配到各个阶段"""
        try:
            template = self.phase_distribution_templates.get(
                strategy, 
                self.phase_distribution_templates[WorkflowStrategy.SYSTEMATIC]
            )
            
            phase_efforts = {}
            remaining_effort = total_effort
            
            # 按比例分配，最后一个阶段承担余数
            phase_names = list(template.keys())
            for i, (phase_name, ratio) in enumerate(template.items()):
                if i == len(phase_names) - 1:
                    # 最后一个阶段承担余数
                    phase_efforts[phase_name] = remaining_effort
                else:
                    effort = int(total_effort * ratio)
                    phase_efforts[phase_name] = effort
                    remaining_effort -= effort
            
            self.logger.info(f"阶段工作量分配完成: {strategy.value} 策略, 总计 {total_effort}h")
            return phase_efforts
            
        except Exception as e:
            self.logger.error(f"阶段工作量分配失败: {str(e)}")
            # 返回平均分配
            phase_count = 4  # 默认4个阶段
            avg_effort = total_effort // phase_count
            return {
                f"阶段{i+1}": avg_effort + (total_effort % phase_count if i == 0 else 0)
                for i in range(phase_count)
            }
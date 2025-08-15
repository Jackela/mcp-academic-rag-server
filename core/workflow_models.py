"""
工作流数据模型

定义工作流生成器使用的核心数据结构和类型。
"""

from typing import List, Dict, Any, Optional, Union, Tuple
from enum import Enum, auto
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import uuid


class WorkflowStrategy(Enum):
    """工作流策略枚举"""
    SYSTEMATIC = "systematic"  # 系统化策略
    AGILE = "agile"           # 敏捷策略  
    MVP = "mvp"               # 最小可行产品策略


class OutputFormat(Enum):
    """输出格式枚举"""
    ROADMAP = "roadmap"       # 路线图格式
    TASKS = "tasks"           # 任务格式
    DETAILED = "detailed"     # 详细格式


class PersonaType(Enum):
    """专家人格类型"""
    FRONTEND = "frontend"
    BACKEND = "backend"
    ARCHITECT = "architect"
    SECURITY = "security"
    DEVOPS = "devops"
    QA = "qa"
    SCRIBE = "scribe"
    PERFORMANCE = "performance"


class Priority(Enum):
    """优先级枚举"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ComplexityLevel(Enum):
    """复杂度级别"""
    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"
    ENTERPRISE = "enterprise"


class RiskLevel(Enum):
    """风险级别"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class Requirement:
    """需求项"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    title: str = ""
    description: str = ""
    type: str = "functional"  # functional, non-functional, constraint
    priority: Priority = Priority.MEDIUM
    complexity: ComplexityLevel = ComplexityLevel.MEDIUM
    estimated_effort: Optional[int] = None  # hours
    acceptance_criteria: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AcceptanceCriteria:
    """验收标准"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    requirement_id: str = ""
    description: str = ""
    is_testable: bool = True
    test_method: str = "manual"  # manual, automated, integration
    success_metric: Optional[str] = None


@dataclass
class PRDStructure:
    """PRD文档结构"""
    title: str = ""
    version: str = "1.0"
    author: str = ""
    date_created: Optional[datetime] = None
    date_updated: Optional[datetime] = None
    overview: str = ""
    objectives: List[str] = field(default_factory=list)
    requirements: List[Requirement] = field(default_factory=list)
    acceptance_criteria: List[AcceptanceCriteria] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    assumptions: List[str] = field(default_factory=list)
    success_metrics: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Dependency:
    """依赖关系"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    type: str = "internal"  # internal, external, technical, team
    description: str = ""
    criticality: Priority = Priority.MEDIUM
    estimated_resolution_time: Optional[int] = None  # hours
    owner: Optional[str] = None
    status: str = "pending"  # pending, in_progress, resolved, blocked
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Risk:
    """风险项"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    category: str = "technical"  # technical, timeline, security, business
    likelihood: RiskLevel = RiskLevel.MEDIUM
    impact: RiskLevel = RiskLevel.MEDIUM
    risk_score: float = 0.0  # calculated from likelihood * impact
    mitigation_strategies: List[str] = field(default_factory=list)
    contingency_plans: List[str] = field(default_factory=list)
    owner: Optional[str] = None
    status: str = "identified"  # identified, mitigating, resolved


@dataclass
class WorkflowStep:
    """工作流步骤"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    estimated_effort: Optional[int] = None  # hours
    complexity: ComplexityLevel = ComplexityLevel.MEDIUM
    persona: Optional[PersonaType] = None
    dependencies: List[str] = field(default_factory=list)
    deliverables: List[str] = field(default_factory=list)
    acceptance_criteria: List[str] = field(default_factory=list)
    tools_required: List[str] = field(default_factory=list)
    code_examples: List[str] = field(default_factory=list)
    mcp_context: Dict[str, str] = field(default_factory=dict)  # server -> context
    order: int = 0


@dataclass
class WorkflowMilestone:
    """工作流里程碑"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    estimated_effort: Optional[int] = None  # hours
    priority: Priority = Priority.MEDIUM
    steps: List[WorkflowStep] = field(default_factory=list)
    success_criteria: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    risks: List[str] = field(default_factory=list)  # risk IDs
    personas_involved: List[PersonaType] = field(default_factory=list)
    order: int = 0


@dataclass
class WorkflowPhase:
    """工作流阶段"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    estimated_duration: Optional[timedelta] = None
    estimated_effort: Optional[int] = None  # total hours
    milestones: List[WorkflowMilestone] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    risks: List[str] = field(default_factory=list)  # risk IDs
    personas_involved: List[PersonaType] = field(default_factory=list)
    parallel_streams: List[str] = field(default_factory=list)  # phase IDs that can run in parallel
    order: int = 0


@dataclass
class ParallelStream:
    """并行工作流"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    phases: List[str] = field(default_factory=list)  # phase IDs
    estimated_effort: Optional[int] = None  # total hours
    required_team_size: int = 1
    coordination_requirements: List[str] = field(default_factory=list)


@dataclass
class CriticalPath:
    """关键路径"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = "Critical Path"
    phases: List[str] = field(default_factory=list)  # ordered phase IDs
    total_duration: Optional[timedelta] = None
    total_effort: Optional[int] = None  # hours
    bottlenecks: List[str] = field(default_factory=list)  # phase/milestone IDs
    optimization_opportunities: List[str] = field(default_factory=list)


@dataclass
class EffortEstimation:
    """工作量估算"""
    total_hours: int = 0
    breakdown_by_persona: Dict[PersonaType, int] = field(default_factory=dict)
    breakdown_by_phase: Dict[str, int] = field(default_factory=dict)  # phase_id -> hours
    confidence_level: float = 0.8  # 0.0 to 1.0
    estimation_method: str = "expert_judgment"  # expert_judgment, historical_data, planning_poker
    buffer_percentage: float = 0.2  # 20% buffer by default
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ComplexityAnalysis:
    """复杂度分析"""
    overall_complexity: ComplexityLevel = ComplexityLevel.MEDIUM
    technical_complexity: float = 0.5  # 0.0 to 1.0
    integration_complexity: float = 0.5
    ui_complexity: float = 0.5
    business_logic_complexity: float = 0.5
    data_complexity: float = 0.5
    complexity_factors: List[str] = field(default_factory=list)
    simplification_opportunities: List[str] = field(default_factory=list)


@dataclass
class RequirementCategories:
    """需求分类"""
    functional_requirements: List[Requirement] = field(default_factory=list)
    non_functional_requirements: List[Requirement] = field(default_factory=list)
    constraint_requirements: List[Requirement] = field(default_factory=list)
    ui_requirements: List[Requirement] = field(default_factory=list)
    api_requirements: List[Requirement] = field(default_factory=list)
    data_requirements: List[Requirement] = field(default_factory=list)
    security_requirements: List[Requirement] = field(default_factory=list)
    performance_requirements: List[Requirement] = field(default_factory=list)


@dataclass
class ProjectConstraints:
    """项目约束条件"""
    timeline: Optional[timedelta] = None
    budget: Optional[float] = None
    team_size: int = 1
    technology_constraints: List[str] = field(default_factory=list)
    compliance_requirements: List[str] = field(default_factory=list)
    performance_requirements: Dict[str, Any] = field(default_factory=dict)
    integration_requirements: List[str] = field(default_factory=list)
    security_requirements: List[str] = field(default_factory=list)


@dataclass
class MCPResults:
    """MCP服务器结果"""
    context7_results: Dict[str, Any] = field(default_factory=dict)
    sequential_results: Dict[str, Any] = field(default_factory=dict)
    magic_results: Dict[str, Any] = field(default_factory=dict)
    playwright_results: Dict[str, Any] = field(default_factory=dict)
    integration_recommendations: List[str] = field(default_factory=list)


@dataclass
class Workflow:
    """完整工作流"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    strategy: WorkflowStrategy = WorkflowStrategy.SYSTEMATIC
    version: str = "1.0"
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    # 需求相关
    prd_structure: Optional[PRDStructure] = None
    requirements: List[Requirement] = field(default_factory=list)
    
    # 工作流结构
    phases: List[WorkflowPhase] = field(default_factory=list)
    parallel_streams: List[ParallelStream] = field(default_factory=list)
    critical_path: Optional[CriticalPath] = None
    
    # 分析结果
    complexity_analysis: Optional[ComplexityAnalysis] = None
    effort_estimation: Optional[EffortEstimation] = None
    requirement_categories: Optional[RequirementCategories] = None
    
    # 依赖和风险
    dependencies: List[Dependency] = field(default_factory=list)
    risks: List[Risk] = field(default_factory=list)
    
    # MCP集成结果
    mcp_results: Optional[MCPResults] = None
    
    # 人格系统
    activated_personas: List[PersonaType] = field(default_factory=list)
    persona_recommendations: Dict[PersonaType, str] = field(default_factory=dict)
    
    # 质量保证
    success_metrics: List[str] = field(default_factory=list)
    quality_gates: List[str] = field(default_factory=list)
    validation_criteria: List[str] = field(default_factory=list)
    
    # 元数据
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_total_effort(self) -> int:
        """获取总工作量"""
        if self.effort_estimation:
            return self.effort_estimation.total_hours
        return sum(phase.estimated_effort or 0 for phase in self.phases)
    
    def get_total_duration(self) -> Optional[timedelta]:
        """获取总持续时间"""
        if self.critical_path and self.critical_path.total_duration:
            return self.critical_path.total_duration
        return None
    
    def get_complexity_score(self) -> float:
        """获取复杂度分数"""
        if self.complexity_analysis:
            return self.complexity_analysis.technical_complexity
        return 0.5
    
    def get_high_risk_items(self) -> List[Risk]:
        """获取高风险项"""
        return [risk for risk in self.risks if risk.likelihood == RiskLevel.HIGH or risk.impact == RiskLevel.HIGH]
    
    def get_critical_dependencies(self) -> List[Dependency]:
        """获取关键依赖"""
        return [dep for dep in self.dependencies if dep.criticality == Priority.HIGH or dep.criticality == Priority.CRITICAL]


@dataclass
class WorkflowOptions:
    """工作流生成选项"""
    strategy: WorkflowStrategy = WorkflowStrategy.SYSTEMATIC
    output_format: OutputFormat = OutputFormat.ROADMAP
    include_estimates: bool = True
    include_dependencies: bool = True
    include_risks: bool = True
    enable_parallel_analysis: bool = True
    enable_milestones: bool = True
    enable_mcp_integration: bool = True
    forced_persona: Optional[PersonaType] = None
    team_size: int = 1
    timeline_constraint: Optional[timedelta] = None
    complexity_preference: Optional[ComplexityLevel] = None
    
    # MCP选项
    enable_context7: bool = False
    enable_sequential: bool = False
    enable_magic: bool = False
    enable_playwright: bool = False
    enable_all_mcp: bool = False
    
    # 输出选项
    include_code_examples: bool = False
    include_templates: bool = False
    include_checklists: bool = True
    include_success_metrics: bool = True
    
    # 高级选项
    enable_optimization_suggestions: bool = True
    enable_quality_gates: bool = True
    enable_continuous_feedback: bool = False
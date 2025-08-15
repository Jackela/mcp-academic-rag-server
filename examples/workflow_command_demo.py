#!/usr/bin/env python3
"""
/sc:workflow 命令演示

展示工作流生成器的完整功能，包括：
1. PRD解析和需求分析
2. 三种策略的工作流生成 (systematic/agile/mvp)
3. 三种输出格式 (roadmap/tasks/detailed)
4. 专家人格激活
5. 依赖分析和风险评估
6. MCP服务集成（模拟）
"""

import asyncio
import sys
import os
import logging
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.workflow_generator import WorkflowGenerator, PRDParser, RequirementAnalyzer
from core.workflow_formatter import WorkflowFormatter
from core.workflow_models import WorkflowStrategy, OutputFormat, WorkflowOptions, PersonaType

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stderr)  # 输出到stderr避免干扰主要输出
    ]
)


# 示例PRD文档内容
SAMPLE_PRD = """
# 用户认证系统 PRD

## 概述
构建一个安全、易用的用户认证系统，支持多种登录方式，包括邮箱密码、社交媒体登录和双因子认证。系统需要具备高安全性、良好的用户体验和可扩展性。

## 项目目标
- 提供安全可靠的用户认证机制
- 支持多种身份验证方式
- 实现用户会话管理
- 确保符合GDPR等隐私法规
- 支持单点登录(SSO)集成

## 功能需求

### 1. 用户注册功能
用户可以通过邮箱地址创建新账户，需要进行邮箱验证。注册过程应该简洁明了，支持密码强度检查。

### 2. 用户登录功能  
支持邮箱+密码登录，集成Google、Facebook等社交媒体登录。登录失败时提供友好的错误提示。

### 3. 双因子认证(2FA)
为增强安全性，提供SMS和TOTP(Time-based One-Time Password)两种双因子认证方式。用户可以选择启用或关闭2FA。

### 4. 密码管理
提供密码重置功能，支持通过邮箱重置密码。实现密码策略管理，要求密码满足复杂度要求。

### 5. 会话管理
实现安全的会话管理机制，支持会话超时、记住登录状态等功能。提供用户主动登出功能。

### 6. 用户Profile管理
用户可以查看和编辑个人信息，包括头像上传、联系方式更新等。支持账户注销功能。

### 7. 管理后台
提供管理员后台，支持用户管理、权限配置、安全日志查看等功能。

## 非功能需求

### 性能要求
- 登录响应时间 < 500ms
- 支持并发用户数 > 10,000
- 系统可用性 > 99.9%

### 安全要求
- 密码加密存储 (bcrypt)
- API接口防护 (rate limiting)
- 数据传输加密 (HTTPS)
- 定期安全审计

### 可扩展性要求
- 支持微服务架构
- 数据库分片支持
- 缓存机制集成

## 验收标准
- 用户可以成功注册并验证邮箱
- 用户可以使用多种方式登录系统
- 双因子认证正常工作
- 密码重置功能完整可用
- 管理后台功能完善
- 通过安全性测试
- 性能指标满足要求

## 约束条件
- 项目周期: 8周
- 团队规模: 5人
- 技术栈: Node.js, React, PostgreSQL
- 预算限制: $50,000

## 风险评估
- 社交媒体API集成的技术风险
- GDPR合规性要求的复杂性
- 高并发场景下的性能挑战
"""

# 简单需求描述示例
SIMPLE_DESCRIPTION = """
实现一个任务管理应用，用户可以创建、编辑、删除任务。
支持任务分类、优先级设置、截止日期提醒功能。
需要提供Web界面和移动端适配。
"""


async def demo_prd_parsing():
    """演示PRD解析功能"""
    print("=" * 80)
    print("🔍 PRD解析演示")
    print("=" * 80)
    
    parser = PRDParser()
    
    print("1. 解析完整PRD文档...")
    prd_structure = parser.parse_content(SAMPLE_PRD, "user_auth_system_prd.md")
    
    print(f"✅ 解析完成:")
    print(f"   标题: {prd_structure.title}")
    print(f"   需求数量: {len(prd_structure.requirements)}")
    print(f"   验收标准: {len(prd_structure.acceptance_criteria)}")
    print(f"   目标数量: {len(prd_structure.objectives)}")
    print(f"   约束条件: {len(prd_structure.constraints)}")
    
    print("\n📋 需求预览:")
    for i, req in enumerate(prd_structure.requirements[:3], 1):
        print(f"   {i}. {req.title} (优先级: {req.priority.value}, 复杂度: {req.complexity.value})")
    
    print(f"\n2. 解析简单文本描述...")
    simple_prd = parser.parse_text_description(SIMPLE_DESCRIPTION)
    
    print(f"✅ 解析完成:")
    print(f"   需求数量: {len(simple_prd.requirements)}")
    
    print("\n📋 简单需求预览:")
    for i, req in enumerate(simple_prd.requirements[:2], 1):
        print(f"   {i}. {req.title}: {req.description[:50]}...")
    
    return prd_structure, simple_prd


async def demo_requirement_analysis():
    """演示需求分析功能"""
    print("\n" + "=" * 80)
    print("📊 需求分析演示") 
    print("=" * 80)
    
    parser = PRDParser()
    analyzer = RequirementAnalyzer()
    
    prd_structure = parser.parse_content(SAMPLE_PRD, "sample_prd")
    requirements = prd_structure.requirements
    
    print("1. 复杂度分析...")
    complexity_analysis = analyzer.analyze_complexity(requirements)
    
    print(f"✅ 分析完成:")
    print(f"   整体复杂度: {complexity_analysis.overall_complexity.value}")
    print(f"   技术复杂度: {complexity_analysis.technical_complexity * 100:.1f}%")
    print(f"   集成复杂度: {complexity_analysis.integration_complexity * 100:.1f}%")
    print(f"   UI复杂度: {complexity_analysis.ui_complexity * 100:.1f}%")
    
    if complexity_analysis.complexity_factors:
        print("   复杂度因素:")
        for factor in complexity_analysis.complexity_factors:
            print(f"     - {factor}")
    
    print("\n2. 需求分类...")
    categories = analyzer.categorize_requirements(requirements)
    
    print(f"✅ 分类完成:")
    print(f"   功能性需求: {len(categories.functional_requirements)}")
    print(f"   非功能性需求: {len(categories.non_functional_requirements)}")  
    print(f"   UI需求: {len(categories.ui_requirements)}")
    print(f"   API需求: {len(categories.api_requirements)}")
    print(f"   安全需求: {len(categories.security_requirements)}")
    print(f"   性能需求: {len(categories.performance_requirements)}")
    
    print("\n3. 工作量估算...")
    estimation = analyzer.estimate_effort(requirements)
    
    print(f"✅ 估算完成:")
    print(f"   总工作量: {estimation.total_hours} 小时")
    print(f"   置信度: {estimation.confidence_level * 100:.0f}%")
    print(f"   缓冲比例: {estimation.buffer_percentage * 100:.0f}%")
    
    if estimation.breakdown_by_persona:
        print("   人员分工:")
        for persona, hours in estimation.breakdown_by_persona.items():
            percentage = (hours / estimation.total_hours) * 100
            print(f"     - {persona.value.title()}: {hours}h ({percentage:.1f}%)")
    
    return complexity_analysis, categories, estimation


async def demo_workflow_generation():
    """演示工作流生成功能"""
    print("\n" + "=" * 80)
    print("🚀 工作流生成演示")
    print("=" * 80)
    
    generator = WorkflowGenerator()
    
    # 生成三种策略的工作流
    strategies = [
        (WorkflowStrategy.SYSTEMATIC, "系统化策略"),
        (WorkflowStrategy.AGILE, "敏捷策略"),
        (WorkflowStrategy.MVP, "MVP策略")
    ]
    
    workflows = {}
    
    for strategy, name in strategies:
        print(f"\n{name} 工作流生成中...")
        
        options = WorkflowOptions(
            strategy=strategy,
            include_estimates=True,
            include_dependencies=True,
            include_risks=True,
            enable_parallel_analysis=True,
            enable_milestones=True,
            enable_mcp_integration=False  # 模拟环境下禁用
        )
        
        workflow = await generator.generate_workflow(SAMPLE_PRD, options)
        workflows[strategy] = workflow
        
        print(f"✅ {name} 生成完成:")
        print(f"   阶段数量: {len(workflow.phases)}")
        print(f"   总工作量: {workflow.get_total_effort()} 小时")
        print(f"   复杂度: {workflow.get_complexity_score() * 100:.1f}%")
        print(f"   激活人格: {', '.join([p.value for p in workflow.activated_personas])}")
        print(f"   依赖项: {len(workflow.dependencies)}")
        print(f"   风险项: {len(workflow.risks)}")
        print(f"   并行流: {len(workflow.parallel_streams)}")
    
    return workflows


async def demo_output_formatting():
    """演示输出格式化功能"""
    print("\n" + "=" * 80)
    print("📄 输出格式化演示")
    print("=" * 80)
    
    generator = WorkflowGenerator() 
    formatter = WorkflowFormatter()
    
    # 生成一个示例工作流
    options = WorkflowOptions(
        strategy=WorkflowStrategy.SYSTEMATIC,
        include_estimates=True,
        include_dependencies=True,
        include_risks=True,
        enable_parallel_analysis=True
    )
    
    workflow = await generator.generate_workflow(SAMPLE_PRD, options)
    
    # 生成三种输出格式
    formats = [
        (OutputFormat.ROADMAP, "路线图格式"),
        (OutputFormat.TASKS, "任务列表格式"),
        (OutputFormat.DETAILED, "详细格式")
    ]
    
    outputs = {}
    
    for output_format, name in formats:
        print(f"\n{name} 生成中...")
        
        formatted_output = formatter.format_workflow(workflow, output_format)
        outputs[output_format] = formatted_output
        
        # 统计输出信息
        lines = formatted_output.split('\n')
        sections = len([line for line in lines if line.startswith('#')])
        tasks = len([line for line in lines if '- [ ]' in line])
        
        print(f"✅ {name} 生成完成:")
        print(f"   总行数: {len(lines)}")
        print(f"   章节数: {sections}")
        print(f"   任务数: {tasks}")
        print(f"   字符数: {len(formatted_output)}")
    
    return workflow, outputs


async def demo_persona_activation():
    """演示专家人格激活"""
    print("\n" + "=" * 80)
    print("👥 专家人格激活演示")
    print("=" * 80)
    
    generator = WorkflowGenerator()
    
    # 测试不同类型需求的人格激活
    test_cases = [
        ("前端重点项目", "开发一个现代化的React单页应用，包含复杂的用户界面组件、状态管理和响应式设计。需要支持多主题切换、国际化和无障碍访问。"),
        ("后端API项目", "构建RESTful API服务，包含用户认证、数据CRUD操作、文件上传下载、第三方服务集成。需要支持高并发访问和数据库优化。"),
        ("企业架构项目", "设计大型电商平台的微服务架构，支持分布式部署、负载均衡、服务发现、配置管理。需要考虑系统扩展性和容错性。"),
        ("安全项目", "实现企业级安全认证系统，包含OAuth2.0、JWT令牌、RBAC权限控制、数据加密、安全审计。需要满足SOC2合规要求。")
    ]
    
    for project_name, description in test_cases:
        print(f"\n{project_name}:")
        print(f"描述: {description[:80]}...")
        
        options = WorkflowOptions(strategy=WorkflowStrategy.SYSTEMATIC)
        workflow = await generator.generate_workflow(description, options)
        
        print(f"激活的专家人格:")
        for persona in workflow.activated_personas:
            print(f"  - {persona.value.title()}: 自动激活")
        
        if workflow.persona_recommendations:
            print(f"专家建议:")
            for persona, recommendation in workflow.persona_recommendations.items():
                print(f"  - {persona.value.title()}: {recommendation}")


async def demo_advanced_features():
    """演示高级功能"""
    print("\n" + "=" * 80)
    print("⚡ 高级功能演示")
    print("=" * 80)
    
    generator = WorkflowGenerator()
    
    # 使用所有高级选项
    options = WorkflowOptions(
        strategy=WorkflowStrategy.SYSTEMATIC,
        output_format=OutputFormat.DETAILED,
        include_estimates=True,
        include_dependencies=True,
        include_risks=True,
        enable_parallel_analysis=True,
        enable_milestones=True,
        enable_mcp_integration=True,  # 启用MCP集成
        forced_persona=PersonaType.ARCHITECT,  # 强制激活架构师
        team_size=5,
        enable_context7=True,
        enable_sequential=True,
        enable_magic=True,
        include_code_examples=True,
        include_templates=True,
        enable_optimization_suggestions=True,
        enable_quality_gates=True
    )
    
    print("生成高级工作流...")
    workflow = await generator.generate_workflow(SAMPLE_PRD, options)
    
    print(f"✅ 高级工作流生成完成:")
    print(f"   策略: {workflow.strategy.value}")
    print(f"   阶段数: {len(workflow.phases)}")
    print(f"   里程碑数: {sum(len(phase.milestones) for phase in workflow.phases)}")
    print(f"   激活人格: {', '.join([p.value for p in workflow.activated_personas])}")
    
    # 并行流分析
    if workflow.parallel_streams:
        print(f"\n🔄 并行工作流分析:")
        for stream in workflow.parallel_streams:
            print(f"   - {stream.name}: {stream.description}")
            if stream.estimated_effort:
                print(f"     工作量: {stream.estimated_effort}h")
            if stream.required_team_size > 1:
                print(f"     团队规模: {stream.required_team_size}人")
    
    # 关键路径分析
    if workflow.critical_path:
        print(f"\n🎯 关键路径分析:")
        print(f"   总工作量: {workflow.critical_path.total_effort}h")
        if workflow.critical_path.total_duration:
            print(f"   预计工期: {workflow.critical_path.total_duration.days}天")
        
        if workflow.critical_path.bottlenecks:
            print(f"   瓶颈数量: {len(workflow.critical_path.bottlenecks)}")
        
        if workflow.critical_path.optimization_opportunities:
            print(f"   优化建议: {len(workflow.critical_path.optimization_opportunities)}个")
    
    # 风险分析
    high_risks = workflow.get_high_risk_items()
    if high_risks:
        print(f"\n⚠️ 高风险项分析:")
        for risk in high_risks[:3]:  # 显示前3个高风险项
            print(f"   - {risk.name}: {risk.description[:50]}...")
            if risk.mitigation_strategies:
                print(f"     缓解策略: {risk.mitigation_strategies[0][:40]}...")
    
    # 关键依赖
    critical_deps = workflow.get_critical_dependencies()
    if critical_deps:
        print(f"\n🔗 关键依赖分析:")
        for dep in critical_deps[:3]:
            print(f"   - {dep.name}: {dep.description[:50]}...")
            print(f"     类型: {dep.type}, 重要性: {dep.criticality.value}")
    
    return workflow


async def save_demo_outputs():
    """保存演示输出到文件"""
    print("\n" + "=" * 80)
    print("💾 保存演示输出")
    print("=" * 80)
    
    generator = WorkflowGenerator()
    formatter = WorkflowFormatter()
    
    # 生成示例工作流
    options = WorkflowOptions(
        strategy=WorkflowStrategy.SYSTEMATIC,
        include_estimates=True,
        include_dependencies=True,
        include_risks=True,
        enable_parallel_analysis=True
    )
    
    workflow = await generator.generate_workflow(SAMPLE_PRD, options)
    
    # 保存不同格式的输出
    output_dir = Path("examples/workflow_outputs")
    output_dir.mkdir(exist_ok=True)
    
    formats = [
        (OutputFormat.ROADMAP, "roadmap.md"),
        (OutputFormat.TASKS, "tasks.md"),
        (OutputFormat.DETAILED, "detailed.md")
    ]
    
    saved_files = []
    
    for output_format, filename in formats:
        formatted_output = formatter.format_workflow(workflow, output_format)
        
        file_path = output_dir / filename
        file_path.write_text(formatted_output, encoding='utf-8')
        
        saved_files.append(str(file_path))
        print(f"✅ 保存 {output_format.value} 格式: {file_path}")
    
    # 保存PRD原文
    prd_file = output_dir / "sample_prd.md"
    prd_file.write_text(SAMPLE_PRD, encoding='utf-8')
    saved_files.append(str(prd_file))
    print(f"✅ 保存PRD原文: {prd_file}")
    
    print(f"\n📁 所有文件已保存到: {output_dir}")
    return saved_files


async def main():
    """主演示函数"""
    print("🎯 /sc:workflow 命令完整演示")
    print("展示SuperClaude工作流生成器的企业级功能")
    print()
    
    try:
        # 1. PRD解析演示
        await demo_prd_parsing()
        
        # 2. 需求分析演示
        await demo_requirement_analysis()
        
        # 3. 工作流生成演示
        await demo_workflow_generation()
        
        # 4. 输出格式化演示
        workflow, outputs = await demo_output_formatting()
        
        # 5. 专家人格激活演示
        await demo_persona_activation()
        
        # 6. 高级功能演示
        advanced_workflow = await demo_advanced_features()
        
        # 7. 保存演示输出
        saved_files = await save_demo_outputs()
        
        # 最终总结
        print("\n" + "=" * 80)
        print("🎉 演示完成总结")
        print("=" * 80)
        
        print("✅ 已演示功能:")
        print("   - PRD文档解析和需求分析")
        print("   - 三种工作流策略 (Systematic/Agile/MVP)")
        print("   - 三种输出格式 (Roadmap/Tasks/Detailed)")
        print("   - 智能专家人格激活")
        print("   - 复杂度和工作量分析")
        print("   - 依赖关系和风险评估")
        print("   - 并行工作流识别")
        print("   - 关键路径计算")
        print("   - 高级功能集成")
        
        print(f"\n📊 核心指标:")
        print(f"   - 解析需求数: {len(workflow.requirements)}")
        print(f"   - 生成阶段数: {len(workflow.phases)}")
        print(f"   - 总工作量: {workflow.get_total_effort()}小时")
        print(f"   - 复杂度得分: {workflow.get_complexity_score() * 100:.1f}%")
        print(f"   - 激活人格数: {len(workflow.activated_personas)}")
        print(f"   - 识别风险数: {len(workflow.risks)}")
        
        print(f"\n💾 输出文件:")
        for file_path in saved_files:
            print(f"   - {file_path}")
        
        print(f"\n🚀 /sc:workflow 命令演示成功!")
        print("企业级工作流生成器已准备就绪，可以处理真实的PRD和项目需求。")
        
    except Exception as e:
        print(f"\n❌ 演示过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
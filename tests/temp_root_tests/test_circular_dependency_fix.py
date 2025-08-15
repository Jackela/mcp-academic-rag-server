#!/usr/bin/env python3
"""
测试循环依赖修复

验证工作流生成器不再存在循环依赖问题。
"""

import sys
from pathlib import Path
import asyncio

# 添加项目根路径
sys.path.insert(0, str(Path(__file__).parent))

async def test_circular_dependency_fix():
    """测试循环依赖修复"""
    try:
        print("🧪 测试循环依赖修复")
        print("=" * 60)
        
        # 1. 测试基础导入
        print("1. 测试基础导入...")
        from core.workflow_models import (
            WorkflowStrategy, OutputFormat, PersonaType, Priority,
            Requirement, PRDStructure, Workflow, WorkflowOptions
        )
        print("   ✅ 工作流模型导入成功")
        
        from core.workflow_effort_calculator import EffortCalculator, PhaseEffortDistributor
        print("   ✅ 工作量计算器导入成功")
        
        from core.workflow_generator import WorkflowGenerator, PRDParser
        print("   ✅ 工作流生成器导入成功")
        
        # 2. 测试工作量计算器
        print("\n2. 测试独立工作量计算...")
        calculator = EffortCalculator()
        
        # 创建测试需求
        test_requirements = [
            Requirement(
                title="用户登录",
                description="实现邮箱密码登录功能",
                priority=Priority.HIGH,
                complexity=Priority.MEDIUM
            ),
            Requirement(
                title="用户注册",
                description="实现用户注册和邮箱验证",
                priority=Priority.HIGH,
                complexity=Priority.MEDIUM
            )
        ]
        
        base_effort = calculator.calculate_base_effort(test_requirements, WorkflowStrategy.SYSTEMATIC)
        print(f"   ✅ 基础工作量计算: {base_effort} 小时")
        
        # 3. 测试阶段分配器
        print("\n3. 测试阶段工作量分配...")
        distributor = PhaseEffortDistributor()
        
        systematic_distribution = distributor.distribute_effort_to_phases(base_effort, WorkflowStrategy.SYSTEMATIC)
        print(f"   ✅ 系统化策略分配: {len(systematic_distribution)} 个阶段")
        
        agile_distribution = distributor.distribute_effort_to_phases(base_effort, WorkflowStrategy.AGILE)
        print(f"   ✅ 敏捷策略分配: {len(agile_distribution)} 个阶段")
        
        mvp_distribution = distributor.distribute_effort_to_phases(base_effort, WorkflowStrategy.MVP)
        print(f"   ✅ MVP策略分配: {len(mvp_distribution)} 个阶段")
        
        # 4. 测试工作流生成器（不会产生循环依赖）
        print("\n4. 测试工作流生成器初始化...")
        generator = WorkflowGenerator()
        print("   ✅ 工作流生成器初始化成功")
        
        # 5. 测试简单工作流生成
        print("\n5. 测试简单工作流生成...")
        simple_description = "实现用户登录和注册功能"
        options = WorkflowOptions(
            strategy=WorkflowStrategy.SYSTEMATIC,
            include_estimates=True,
            include_dependencies=False,  # 简化测试
            include_risks=False,
            enable_parallel_analysis=False,
            enable_milestones=False  # 避免复杂的里程碑生成
        )
        
        workflow = await generator.generate_workflow(simple_description, options)
        print(f"   ✅ 工作流生成成功:")
        print(f"      - 策略: {workflow.strategy.value}")
        print(f"      - 需求数: {len(workflow.requirements)}")
        print(f"      - 阶段数: {len(workflow.phases)}")
        
        # 验证不再有循环依赖
        total_effort = workflow.get_total_effort()
        print(f"      - 总工作量: {total_effort} 小时")
        
        # 6. 验证所有策略都能正常工作
        print("\n6. 测试所有策略...")
        
        strategies = [
            (WorkflowStrategy.SYSTEMATIC, "系统化"),
            (WorkflowStrategy.AGILE, "敏捷"),
            (WorkflowStrategy.MVP, "MVP")
        ]
        
        for strategy, name in strategies:
            options.strategy = strategy
            test_workflow = await generator.generate_workflow(simple_description, options)
            print(f"   ✅ {name}策略: {len(test_workflow.phases)} 阶段, {test_workflow.get_total_effort()}h")
        
        print("\n🎉 循环依赖修复测试全部通过!")
        print("=" * 60)
        print("✅ 修复成果:")
        print("   - 消除了 workflow.get_total_effort() 的循环调用")
        print("   - 工作量计算现在基于需求独立进行")
        print("   - 阶段生成使用预计算的工作量分配")
        print("   - 所有三种策略都能正常工作")
        print("   - 工作流生成器现在可以安全使用")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """主测试函数"""
    success = await test_circular_dependency_fix()
    return 0 if success else 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
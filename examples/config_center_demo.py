#!/usr/bin/env python3
"""
ConfigCenter 使用演示

展示统一配置中心的核心功能：
- 多环境配置支持
- 配置热更新
- 变更监听
- 配置验证
- 备份和恢复
"""

import asyncio
import logging
import os
import sys
import time
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.config_center import ConfigCenter, ConfigChangeEvent, get_config_center, init_config_center

# 配置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


def on_config_change(event: ConfigChangeEvent):
    """配置变更回调函数"""
    print(f"🔄 配置变更: {event.key}")
    print(f"   旧值: {event.old_value}")
    print(f"   新值: {event.new_value}")
    print(f"   时间: {event.timestamp}")
    print("-" * 50)


def demo_basic_usage():
    """基本使用演示"""
    print("=" * 60)
    print("🚀 ConfigCenter 基本使用演示")
    print("=" * 60)

    # 初始化配置中心
    config_center = init_config_center(base_config_path="./config", environment="development", watch_changes=True)

    # 添加变更监听器
    config_center.add_change_listener(on_config_change)

    # 获取配置值
    print("📋 当前配置信息:")
    print(f"服务器名称: {config_center.get_value('server.name')}")
    print(f"服务器端口: {config_center.get_value('server.port')}")
    print(f"向量存储类型: {config_center.get_value('vector_db.type')}")
    print(f"LLM提供商: {config_center.get_value('llm.provider')}")

    # 获取统计信息
    print("\n📊 配置中心统计:")
    stats = config_center.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    return config_center


def demo_environment_switching():
    """环境切换演示"""
    print("\n" + "=" * 60)
    print("🔄 环境切换演示")
    print("=" * 60)

    config_center = get_config_center()

    print(f"📍 当前环境: {config_center.environment}")
    print(f"当前端口: {config_center.get_value('server.port')}")

    # 切换到生产环境
    print("\n🔄 切换到生产环境...")
    if config_center.switch_environment("production"):
        print(f"✅ 环境切换成功!")
        print(f"新环境: {config_center.environment}")
        print(f"新端口: {config_center.get_value('server.port')}")
        print(f"新LLM模型: {config_center.get_value('llm.model')}")

    # 切换回开发环境
    print("\n🔄 切换回开发环境...")
    if config_center.switch_environment("development"):
        print(f"✅ 环境切换成功!")
        print(f"环境: {config_center.environment}")
        print(f"端口: {config_center.get_value('server.port')}")


def demo_runtime_config_changes():
    """运行时配置修改演示"""
    print("\n" + "=" * 60)
    print("⚙️ 运行时配置修改演示")
    print("=" * 60)

    config_center = get_config_center()

    print("📋 原始配置:")
    print(f"服务器端口: {config_center.get_value('server.port')}")
    print(f"日志级别: {config_center.get_value('logging.level')}")

    # 修改配置
    print("\n🔧 修改配置...")
    config_center.set_value("server.port", 9000)
    config_center.set_value("logging.level", "WARNING")

    print("\n📋 修改后配置:")
    print(f"服务器端口: {config_center.get_value('server.port')}")
    print(f"日志级别: {config_center.get_value('logging.level')}")

    # 恢复原始配置
    print("\n🔄 恢复原始配置...")
    config_center.set_value("server.port", 8001)
    config_center.set_value("logging.level", "DEBUG")


def demo_config_validation():
    """配置验证演示"""
    print("\n" + "=" * 60)
    print("✅ 配置验证演示")
    print("=" * 60)

    config_center = get_config_center()

    # 验证当前配置
    validation_result = config_center.validate_current_config()
    print(f"📋 配置验证结果: {'通过' if validation_result['is_valid'] else '失败'}")

    if not validation_result["is_valid"]:
        report = validation_result["report"]
        print(f"❌ 错误: {report.get('errors', [])}")
        print(f"⚠️ 警告: {report.get('warnings', [])}")

    # 尝试设置无效配置
    print("\n🧪 尝试设置无效配置...")
    success = config_center.set_value("server.port", "invalid_port")
    print(f"设置结果: {'成功' if success else '失败(配置验证阻止)'}")


def demo_backup_restore():
    """备份和恢复演示"""
    print("\n" + "=" * 60)
    print("💾 备份和恢复演示")
    print("=" * 60)

    config_center = get_config_center()

    # 创建备份
    print("📁 创建配置备份...")
    try:
        backup_path = config_center.backup_config()
        print(f"✅ 备份已创建: {backup_path}")

        # 修改配置
        print("\n🔧 修改配置...")
        original_port = config_center.get_value("server.port")
        config_center.set_value("server.port", 7777)
        print(f"端口已修改为: {config_center.get_value('server.port')}")

        # 等待一下
        time.sleep(1)

        # 恢复配置
        print("\n🔄 从备份恢复配置...")
        if config_center.restore_config(backup_path):
            print("✅ 配置恢复成功!")
            print(f"恢复后端口: {config_center.get_value('server.port')}")

        # 清理备份文件
        if os.path.exists(backup_path):
            os.remove(backup_path)
            print(f"🗑️ 已清理备份文件: {backup_path}")

    except Exception as e:
        print(f"❌ 备份演示失败: {str(e)}")


def demo_advanced_features():
    """高级功能演示"""
    print("\n" + "=" * 60)
    print("🎯 高级功能演示")
    print("=" * 60)

    config_center = get_config_center()

    # 获取环境配置
    print("📂 可用环境配置:")
    for env in ["default", "development", "production"]:
        env_config = config_center.get_environment_config(env)
        if env_config:
            print(f"  ✅ {env}: {len(env_config)} 个配置项")
        else:
            print(f"  ❌ {env}: 不可用")

    # 配置统计
    print(f"\n📊 详细统计信息:")
    stats = config_center.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")


async def demo_async_monitoring():
    """异步监控演示"""
    print("\n" + "=" * 60)
    print("🔄 异步配置监控演示")
    print("=" * 60)

    config_center = get_config_center()

    # 模拟异步配置变更
    print("🚀 启动异步配置监控...")

    async def config_monitor():
        """异步配置监控任务"""
        for i in range(3):
            await asyncio.sleep(2)
            new_port = 8000 + i + 10
            config_center.set_value(f"test.async_port_{i}", new_port)
            print(f"⏰ 异步设置 test.async_port_{i} = {new_port}")

    # 运行异步任务
    await config_monitor()
    print("✅ 异步监控演示完成")


def main():
    """主函数"""
    try:
        # 基本使用演示
        config_center = demo_basic_usage()

        # 环境切换演示
        demo_environment_switching()

        # 运行时配置修改
        demo_runtime_config_changes()

        # 配置验证
        demo_config_validation()

        # 备份恢复
        demo_backup_restore()

        # 高级功能
        demo_advanced_features()

        # 异步监控
        print("\n🔄 运行异步监控演示...")
        asyncio.run(demo_async_monitoring())

        # 最终统计
        print("\n" + "=" * 60)
        print("📊 最终统计信息")
        print("=" * 60)
        final_stats = config_center.get_stats()
        for key, value in final_stats.items():
            print(f"  {key}: {value}")

        print(f"\n🎉 ConfigCenter 演示完成!")
        print(f"✅ 总变更次数: {final_stats['total_changes']}")
        print(f"✅ 总重载次数: {final_stats['total_reloads']}")

        # 关闭配置中心
        config_center.close()

    except KeyboardInterrupt:
        print("\n⏹️ 用户中断演示")
    except Exception as e:
        print(f"\n❌ 演示过程中出现错误: {str(e)}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()

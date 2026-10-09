#!/usr/bin/env python3
"""
配置验证工具

用于验证MCP Academic RAG Server的配置文件，检查格式错误、缺失字段和不一致问题。
"""

import argparse
import json
import os
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from core.config_manager import ConfigManager
from core.config_validator import ConfigValidator, generate_default_config, validate_config_file


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="MCP Academic RAG Server 配置验证工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python validate_config.py                           # 验证默认配置文件
  python validate_config.py -c ./config/config.json  # 验证指定配置文件
  python validate_config.py --fix                     # 尝试修复配置问题
  python validate_config.py --generate-default        # 生成默认配置文件
  python validate_config.py --report                  # 生成详细验证报告
        """,
    )

    parser.add_argument(
        "-c", "--config", default="./config/config.json", help="配置文件路径 (默认: ./config/config.json)"
    )

    parser.add_argument("--fix", action="store_true", help="尝试自动修复配置问题")

    parser.add_argument("--generate-default", action="store_true", help="生成默认配置文件")

    parser.add_argument("--report", action="store_true", help="生成详细验证报告")

    parser.add_argument("--output", help="输出文件路径（用于生成配置或报告）")

    parser.add_argument("--verbose", "-v", action="store_true", help="详细输出")

    args = parser.parse_args()

    # 生成默认配置
    if args.generate_default:
        generate_default_config_file(args.output or "./config/config_default.json")
        return

    # 验证配置文件
    config_path = args.config

    if not os.path.exists(config_path):
        print(f"❌ 错误: 配置文件不存在: {config_path}")
        sys.exit(1)

    print(f"🔍 验证配置文件: {config_path}")
    print("=" * 60)

    # 执行验证
    is_valid, report = validate_config_file(config_path)

    # 输出验证结果
    if is_valid:
        print("✅ 配置验证通过！")
    else:
        print("❌ 配置验证失败！")

        if report.get("errors"):
            print("\n🚨 错误:")
            for error in report["errors"]:
                print(f"  • {error}")

    # 输出警告
    if report.get("warnings"):
        print("\n⚠️  警告:")
        for warning in report["warnings"]:
            print(f"  • {warning}")

    # 生成详细报告
    if args.report:
        generate_validation_report(config_path, report, args.output)

    # 尝试修复配置
    if args.fix and not is_valid:
        print("\n🔧 尝试修复配置问题...")
        fix_config_issues(config_path)

    # 输出详细信息
    if args.verbose:
        print_verbose_info(config_path)

    # 设置退出码
    sys.exit(0 if is_valid else 1)


def generate_default_config_file(output_path: str):
    """生成默认配置文件"""
    try:
        default_config = generate_default_config()

        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(default_config, f, indent=2, ensure_ascii=False)

        print(f"✅ 默认配置已生成: {output_path}")

    except Exception as e:
        print(f"❌ 生成默认配置失败: {str(e)}")
        sys.exit(1)


def generate_validation_report(config_path: str, report: dict, output_path: str = None):
    """生成验证报告"""
    if not output_path:
        output_path = config_path.replace(".json", "_validation_report.txt")

    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("MCP Academic RAG Server 配置验证报告\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"配置文件: {config_path}\n")
            f.write(f"验证时间: {__import__('datetime').datetime.now().isoformat()}\n\n")

            # 验证结果
            f.write("验证结果:\n")
            f.write("-" * 20 + "\n")
            if report.get("is_valid", len(report.get("errors", [])) == 0):
                f.write("✅ 配置有效\n\n")
            else:
                f.write("❌ 配置无效\n\n")

            # 错误列表
            if report.get("errors"):
                f.write("错误列表:\n")
                f.write("-" * 20 + "\n")
                for i, error in enumerate(report["errors"], 1):
                    f.write(f"{i}. {error}\n")
                f.write("\n")

            # 警告列表
            if report.get("warnings"):
                f.write("警告列表:\n")
                f.write("-" * 20 + "\n")
                for i, warning in enumerate(report["warnings"], 1):
                    f.write(f"{i}. {warning}\n")
                f.write("\n")

            # 建议
            f.write("建议:\n")
            f.write("-" * 20 + "\n")
            if report.get("errors"):
                f.write("• 请修复上述错误后重新验证配置\n")
                f.write("• 可以使用 --fix 参数尝试自动修复\n")
            if report.get("warnings"):
                f.write("• 请考虑解决上述警告以获得最佳性能\n")
            f.write("• 参考文档了解详细配置说明\n")

        print(f"📄 验证报告已生成: {output_path}")

    except Exception as e:
        print(f"❌ 生成验证报告失败: {str(e)}")


def fix_config_issues(config_path: str):
    """尝试修复配置问题"""
    try:
        config_manager = ConfigManager(config_path)

        if config_manager.fix_config_issues():
            print("✅ 配置问题已修复，请重新验证")
        else:
            print("❌ 无法自动修复配置问题，请手动检查")

    except Exception as e:
        print(f"❌ 修复配置时发生错误: {str(e)}")


def print_verbose_info(config_path: str):
    """输出详细信息"""
    try:
        config_manager = ConfigManager(config_path)
        config = config_manager.get_config()

        print("\n📊 配置统计:")
        print("-" * 30)

        # 处理器统计
        processors = config.get("processors", {})
        enabled_processors = [name for name, cfg in processors.items() if cfg.get("enabled", False)]
        print(f"处理器总数: {len(processors)}")
        print(f"启用的处理器: {len(enabled_processors)}")
        if enabled_processors:
            print(f"启用列表: {', '.join(enabled_processors)}")

        # 连接器统计
        connectors = config.get("connectors", {})
        print(f"连接器总数: {len(connectors)}")

        # RAG设置
        rag_settings = config.get("rag_settings", {})
        if rag_settings:
            print(f"RAG Top-K: {rag_settings.get('top_k', 'N/A')}")
            print(f"RAG 阈值: {rag_settings.get('threshold', 'N/A')}")

        # 存储路径
        storage = config.get("storage", {})
        if storage:
            print(f"数据路径: {storage.get('base_path', 'N/A')}")
            print(f"输出路径: {storage.get('output_path', 'N/A')}")

    except Exception as e:
        print(f"❌ 获取详细信息失败: {str(e)}")


if __name__ == "__main__":
    main()

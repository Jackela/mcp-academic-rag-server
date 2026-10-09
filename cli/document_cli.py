"""
文档处理命令行界面

该模块提供命令行界面，用于文档上传、处理和查询功能。
"""

import argparse
import asyncio
import glob
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# 添加项目根目录到系统路径，确保能够导入其他模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config_manager import ConfigManager  # noqa: E402 - direct script entry point
from core.pipeline import Pipeline  # noqa: E402 - direct script entry point
from core.processor_loader import ProcessorLoader  # noqa: E402 - direct script entry point
from models.document import Document  # noqa: E402 - direct script entry point


class DocumentCLI:
    """文档处理命令行界面类"""

    def __init__(self, config_path: str = "./config/config.json", verbose: bool = False) -> None:
        """
        初始化文档处理命令行界面

        Args:
            config_path (str): 配置文件路径
            verbose (bool): 是否显示详细日志
        """
        self.config_path = config_path
        self.verbose = verbose

        # 设置日志级别
        self.log_level = logging.DEBUG if verbose else logging.INFO

        # 初始化组件
        self._init_components()

    def _init_components(self) -> None:
        """初始化组件：配置管理器、处理流水线等"""
        try:
            # 初始化配置管理器
            self.config_manager = ConfigManager(self.config_path)

            # 设置日志
            self._setup_logging()

            # 创建存储目录
            self._create_storage_dirs()

            # 记录初始化信息
            self.logger.info(f"文档处理CLI初始化完成，配置文件：{self.config_path}")

            # 加载处理器（实际应用中应实现动态加载）
            self.processors_loaded = False

            # 初始化处理流水线
            self.pipeline = Pipeline("DocumentCLI_Pipeline")

        except Exception as e:
            print(f"初始化文档处理CLI失败: {str(e)}")
            sys.exit(1)

    def _setup_logging(self) -> None:
        """设置日志系统"""
        log_config = self.config_manager.get_value("logging", {})
        log_level_name = log_config.get("level", "INFO")

        if self.verbose:
            log_level_name = "DEBUG"

        log_level = getattr(logging, log_level_name)
        log_format = log_config.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        log_file = log_config.get("file")

        # 创建日志目录
        if log_file:
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)

        # 配置根日志记录器
        logging.basicConfig(
            level=log_level,
            format=log_format,
            handlers=[
                logging.FileHandler(log_file, encoding="utf-8") if log_file else logging.NullHandler(),
                logging.StreamHandler(),
            ],
        )

        # 获取日志记录器
        self.logger = logging.getLogger("document_cli")

    def _create_storage_dirs(self) -> None:
        """创建存储目录"""
        storage_base_path = self.config_manager.get_value("storage.base_path", "./data")
        storage_output_path = self.config_manager.get_value("storage.output_path", "./output")

        os.makedirs(storage_base_path, exist_ok=True)
        os.makedirs(storage_output_path, exist_ok=True)

        self.storage_base_path = storage_base_path
        self.storage_output_path = storage_output_path

    def _parse_args(self) -> argparse.Namespace:
        """解析命令行参数"""
        parser = argparse.ArgumentParser(
            description="文档处理命令行界面 - 提供文档上传、处理和查询功能",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
示例用法:
  # 上传并处理单个文档
  python document_cli.py upload --file path/to/document.pdf

  # 批量上传并处理文档
  python document_cli.py upload --directory path/to/documents

  # 使用特定处理器处理文档
  python document_cli.py process --id document_id --processors OCRProcessor,StructureProcessor

  # 查询文档信息
  python document_cli.py info --id document_id

  # 列出所有已处理的文档
  python document_cli.py list

  # 导出处理后的文档
  python document_cli.py export --id document_id --format markdown
            """,
        )

        # 创建子命令
        subparsers = parser.add_subparsers(dest="command", help="子命令")

        # upload命令
        upload_parser = subparsers.add_parser("upload", help="上传并处理文档")
        upload_group = upload_parser.add_mutually_exclusive_group(required=True)
        upload_group.add_argument("--file", help="要上传的文件路径")
        upload_group.add_argument("--directory", help="要上传的文档目录")
        upload_parser.add_argument("--recursive", action="store_true", help="递归处理目录中的文件")
        upload_parser.add_argument("--extensions", help="要处理的文件扩展名，逗号分隔，例如：pdf,jpg,png")

        # process命令
        process_parser = subparsers.add_parser("process", help="处理已上传的文档")
        process_parser.add_argument("--id", required=True, help="文档ID")
        process_parser.add_argument(
            "--processors", help="要使用的处理器，逗号分隔，例如：OCRProcessor,StructureProcessor"
        )

        # info命令
        info_parser = subparsers.add_parser("info", help="查询文档信息")
        info_parser.add_argument("--id", required=True, help="文档ID")

        # list命令
        list_parser = subparsers.add_parser("list", help="列出所有已处理的文档")
        list_parser.add_argument("--status", help="按状态筛选文档，例如：completed,error")
        list_parser.add_argument("--tag", help="按标签筛选文档")
        list_parser.add_argument("--format", choices=["table", "json"], default="table", help="输出格式：表格或JSON")

        # export命令
        export_parser = subparsers.add_parser("export", help="导出处理后的文档")
        export_parser.add_argument("--id", required=True, help="文档ID")
        export_parser.add_argument("--format", choices=["markdown", "pdf", "text"], default="markdown", help="导出格式")
        export_parser.add_argument("--output", help="输出文件路径")

        # delete命令
        delete_parser = subparsers.add_parser("delete", help="删除文档")
        delete_parser.add_argument("--id", required=True, help="文档ID")
        delete_parser.add_argument("--confirm", action="store_true", help="确认删除，不提示")

        # 全局选项
        parser.add_argument("--config", default=self.config_path, help="配置文件路径")
        parser.add_argument("--verbose", "-v", action="store_true", help="显示详细日志")

        return parser.parse_args()

    def run(self) -> None:
        """运行命令行界面"""
        args = self._parse_args()

        # 更新配置文件路径
        if args.config != self.config_path:
            self.config_path = args.config
            self.config_manager = ConfigManager(self.config_path)
            self._create_storage_dirs()
            self.processors_loaded = False
            self.pipeline.clear_processors()

        # 更新日志级别
        if args.verbose:
            self.verbose = True
            logging.getLogger().setLevel(logging.DEBUG)
            self.logger.setLevel(logging.DEBUG)
            self.logger.debug("已启用详细日志模式")

        # 处理命令
        if args.command == "upload":
            self._handle_upload(args)
        elif args.command == "process":
            self._handle_process(args)
        elif args.command == "info":
            self._handle_info(args)
        elif args.command == "list":
            self._handle_list(args)
        elif args.command == "export":
            self._handle_export(args)
        elif args.command == "delete":
            self._handle_delete(args)
        else:
            # 没有提供子命令，显示帮助
            self._print_usage()

    def _handle_upload(self, args: argparse.Namespace) -> None:
        """处理上传命令"""
        self.logger.info(f"处理上传命令: {args}")

        for file_path in self._upload_paths(args):
            document = Document(file_path)
            self._process_document(document)

    def _upload_paths(self, args: argparse.Namespace) -> list[str]:
        if args.file:
            if os.path.isfile(args.file):
                return [args.file]
            self.logger.error(f"文件不存在：{args.file}")
            return []
        if not args.directory or not os.path.isdir(args.directory):
            self.logger.error(f"目录不存在：{args.directory}")
            return []
        extensions = [".pdf", ".jpg", ".png", ".tiff", ".jpeg", ".bmp"]
        if args.extensions:
            extensions = ["." + extension.lower().strip() for extension in args.extensions.split(",")]
        pattern = os.path.join(args.directory, "**", "*") if args.recursive else os.path.join(args.directory, "*")
        return [
            path
            for path in glob.glob(pattern, recursive=bool(args.recursive))
            if os.path.isfile(path) and os.path.splitext(path)[1].lower() in extensions
        ]

    def _process_document(self, document: Document, processor_names: list[str] | None = None) -> None:
        """Run the configured pipeline and persist its actual result and content."""
        if not self.processors_loaded:
            for processor in ProcessorLoader(self.config_manager).load_processors():
                self.pipeline.add_processor(processor)
            self.processors_loaded = True
        original = self.pipeline.get_processors()
        try:
            if processor_names:
                if not self.pipeline.reorder_processors(processor_names):
                    raise ValueError("Requested processors are unavailable")
            result = asyncio.run(self.pipeline.process_document(document))
            if not result.is_successful():
                document.update_status("error")
                print(f"处理失败：{result.get_message()}")
            else:
                print(f"文档ID：{document.document_id}，状态：{document.status}")
        except Exception as error:
            document.update_status("error")
            print(f"处理失败：{error}")
        finally:
            self.pipeline.clear_processors()
            for processor in original:
                self.pipeline.add_processor(processor)
            directory = Path(self.storage_base_path) / document.document_id
            directory.mkdir(parents=True, exist_ok=True)
            data = document.to_dict()
            data["content"] = document.content
            with (directory / "document.json").open("w", encoding="utf-8") as saved:
                json.dump(data, saved, ensure_ascii=False, indent=2, default=self._serialize_document_value)

    @staticmethod
    def _serialize_document_value(value: object) -> str:
        if isinstance(value, datetime):
            return value.isoformat()
        if isinstance(value, Path):
            return str(value)
        raise TypeError(f"Cannot persist document value of type {type(value).__name__}")

    @staticmethod
    def _restore_document(data: dict[str, Any]) -> Document:
        document = Document.from_dict(data)
        document.content = data.get("content", {})
        return document

    def _handle_process(self, args: argparse.Namespace) -> None:
        """处理process命令"""
        self.logger.info(f"处理process命令: {args}")

        document_id = args.id
        document_dir = os.path.join(self.storage_base_path, document_id)
        document_info_path = os.path.join(document_dir, "document.json")

        # 检查文档是否存在
        if not os.path.exists(document_info_path):
            self.logger.error(f"文档不存在：{document_id}")
            print(f"错误：文档ID {document_id} 不存在")
            return

        # 加载文档信息
        try:
            with open(document_info_path, "r", encoding="utf-8") as f:
                document_data = json.load(f)

            # 创建Document对象
            document = self._restore_document(document_data)
            self.logger.info(f"已加载文档：{document_id}")

            names = [name.strip() for name in args.processors.split(",")] if args.processors else None
            self._process_document(document, names)

        except Exception as e:
            self.logger.error(f"处理文档 {document_id} 时出错: {str(e)}")
            print(f"错误：处理文档时出错：{str(e)}")

    def _handle_info(self, args: argparse.Namespace) -> None:
        """处理info命令"""
        self.logger.info(f"处理info命令: {args}")

        document_id = args.id
        document_dir = os.path.join(self.storage_base_path, document_id)
        document_info_path = os.path.join(document_dir, "document.json")

        # 检查文档是否存在
        if not os.path.exists(document_info_path):
            self.logger.error(f"文档不存在：{document_id}")
            print(f"错误：文档ID {document_id} 不存在")
            return

        # 加载文档信息
        try:
            with open(document_info_path, "r", encoding="utf-8") as f:
                document_data = json.load(f)

            # 创建Document对象
            document = self._restore_document(document_data)

            # 显示文档信息
            print(f"文档ID: {document.document_id}")
            print(f"文件名: {document.file_name}")
            print(f"创建时间: {document.creation_time}")
            print(f"修改时间: {document.modification_time}")
            print(f"状态: {document.status}")
            print(f"元数据: {document.metadata}")
            print(f"标签: {document.tags}")
            print(f"处理历史: {document.processing_history}")

            self.logger.info(f"显示了文档 {document_id} 的信息")

        except Exception as e:
            self.logger.error(f"获取文档 {document_id} 信息时出错: {str(e)}")
            print(f"错误：获取文档信息时出错：{str(e)}")

    def _handle_list(self, args: argparse.Namespace) -> None:
        """处理list命令"""
        self.logger.info(f"处理list命令: {args}")

        # 获取过滤条件
        status_filter = args.status.split(",") if args.status else None
        tag_filter = args.tag
        output_format = args.format

        # 查找所有文档
        documents = []

        try:
            # 遍历存储目录
            for document_id in os.listdir(self.storage_base_path):
                document_dir = os.path.join(self.storage_base_path, document_id)
                document_info_path = os.path.join(document_dir, "document.json")

                # 检查是否是有效的文档目录
                if os.path.isfile(document_info_path):
                    try:
                        with open(document_info_path, "r", encoding="utf-8") as f:
                            document_data = json.load(f)

                        if not self._matches_document(document_data, status_filter, tag_filter):
                            continue

                        documents.append(document_data)
                    except Exception as e:
                        self.logger.warning(f"读取文档 {document_id} 信息时出错: {str(e)}")

            # 按修改时间排序
            documents.sort(key=lambda x: x.get("modification_time", ""), reverse=True)

            # 输出结果
            if output_format == "json":
                print(json.dumps(documents, indent=2, ensure_ascii=False))
            else:  # 表格格式
                if not documents:
                    print("没有找到文档")
                    return

                self._print_document_rows(documents)

            self.logger.info(f"列出了 {len(documents)} 个文档")

        except Exception as e:
            self.logger.error(f"列出文档时出错: {str(e)}")
            print(f"错误：列出文档时出错：{str(e)}")

    @staticmethod
    def _matches_document(data: dict[str, Any], statuses: list[str] | None, tag: str | None) -> bool:
        return (not statuses or data.get("status") in statuses) and (not tag or tag in data.get("tags", []))

    @staticmethod
    def _print_document_rows(documents: list[dict[str, Any]]) -> None:
        # 打印表头
        print(f"{'文档ID':<36} | {'文件名':<20} | {'状态':<10} | {'创建时间':<20} | {'标签':<20}")
        print("-" * 120)

        # 打印每个文档
        for doc in documents:
            tags = ", ".join(doc.get("tags", []))[:20] if doc.get("tags") else "[]"
            document_id = doc.get("document_id", "N/A")
            file_name = doc.get("file_name", "N/A")[:20]
            status = doc.get("status", "N/A")
            creation_time = doc.get("creation_time", "N/A")[:20]
            print(f"{document_id:<36} | {file_name:<20} | {status:<10} | {creation_time:<20} | {tags:<20}")

    def _handle_export(self, args: argparse.Namespace) -> None:
        """处理export命令"""
        self.logger.info(f"处理export命令: {args}")

        document_id = args.id
        export_format = args.format
        document_dir = os.path.join(self.storage_base_path, document_id)
        document_info_path = os.path.join(document_dir, "document.json")

        # 检查文档是否存在
        if not os.path.exists(document_info_path):
            self.logger.error(f"文档不存在：{document_id}")
            print(f"错误：文档ID {document_id} 不存在")
            return

        # 加载文档信息
        try:
            with open(document_info_path, "r", encoding="utf-8") as f:
                document_data = json.load(f)

            # 创建Document对象
            document = self._restore_document(document_data)

            # 确定输出文件路径
            output_path = args.output
            if not output_path:
                # 如果未指定输出路径，使用默认路径
                filename = f"{document.file_name.rsplit('.', 1)[0]}.{export_format}"
                output_path = os.path.join(self.storage_output_path, filename)

            # 检查输出目录是否存在
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)

            # 导出文档
            print(f"导出文档：{document_id}")
            print(f"  格式：{export_format}")
            print(f"  输出路径：{output_path}")

            text = document.get_text_content()
            if not text:
                raise ValueError("文档没有可导出的实际文本内容")
            if export_format == "pdf":
                raise ValueError("PDF导出尚未实现，请选择markdown或text")
            with open(output_path, "w", encoding="utf-8") as exported:
                exported.write(text)
            print("  导出完成！")

            self.logger.info(f"已将文档 {document_id} 导出为 {export_format} 格式：{output_path}")

        except Exception as e:
            self.logger.error(f"导出文档 {document_id} 时出错: {str(e)}")
            print(f"错误：导出文档时出错：{str(e)}")

    def _handle_delete(self, args: argparse.Namespace) -> None:
        """处理delete命令"""
        self.logger.info(f"处理delete命令: {args}")

        document_id = args.id
        document_dir = os.path.join(self.storage_base_path, document_id)

        # 检查文档是否存在
        if not os.path.exists(document_dir):
            self.logger.error(f"文档不存在：{document_id}")
            print(f"错误：文档ID {document_id} 不存在")
            return

        # 确认删除
        if not args.confirm:
            confirm = input(f"确定要删除文档 {document_id} 吗？此操作不可撤销。(y/n): ")
            if confirm.lower() != "y":
                print("已取消删除操作")
                return

        # 删除文档
        try:
            import shutil

            shutil.rmtree(document_dir)

            print(f"已删除文档：{document_id}")
            self.logger.info(f"已删除文档 {document_id}")

        except Exception as e:
            self.logger.error(f"删除文档 {document_id} 时出错: {str(e)}")
            print(f"错误：删除文档时出错：{str(e)}")

    def _print_usage(self) -> None:
        """打印使用说明"""
        # 通过创建解析器并打印帮助来显示使用说明
        parser = argparse.ArgumentParser(description="文档处理命令行界面 - 提供文档上传、处理和查询功能")
        subparsers = parser.add_subparsers(dest="command", help="子命令")

        # upload命令
        subparsers.add_parser("upload", help="上传并处理文档")

        # process命令
        subparsers.add_parser("process", help="处理已上传的文档")

        # info命令
        subparsers.add_parser("info", help="查询文档信息")

        # list命令
        subparsers.add_parser("list", help="列出所有已处理的文档")

        # export命令
        subparsers.add_parser("export", help="导出处理后的文档")

        # delete命令
        subparsers.add_parser("delete", help="删除文档")

        # 全局选项
        parser.add_argument("--config", help="配置文件路径")
        parser.add_argument("--verbose", "-v", action="store_true", help="显示详细日志")

        # 打印帮助信息
        parser.print_help()


def main() -> None:
    """主入口函数"""
    try:
        cli = DocumentCLI()
        cli.run()
    except KeyboardInterrupt:
        print("\n操作已取消")
        sys.exit(1)
    except Exception as e:
        print(f"错误：{str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()

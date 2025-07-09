"""
文档处理命令行界面

该模块提供命令行界面，用于文档上传、处理和查询功能。
"""

import os
import sys
import argparse
import logging
from typing import List, Dict, Any, Optional
import json
import glob
import time
from pathlib import Path

# 添加项目根目录到系统路径，确保能够导入其他模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config_manager import ConfigManager
from core.pipeline import Pipeline
from models.document import Document
from models.process_result import ProcessResult
from processors.base_processor import IProcessor


class DocumentCLI:
    """文档处理命令行界面类"""
    
    def __init__(
        self,
        config_path: str = "./config/config.json",
        verbose: bool = False
    ):
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
    
    def _init_components(self):
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
            self.processors = {}
            
            # 初始化处理流水线
            self.pipeline = Pipeline("DocumentCLI_Pipeline")
            
        except Exception as e:
            print(f"初始化文档处理CLI失败: {str(e)}")
            sys.exit(1)
    
    def _setup_logging(self):
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
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        # 配置根日志记录器
        logging.basicConfig(
            level=log_level,
            format=log_format,
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8') if log_file else logging.NullHandler(),
                logging.StreamHandler()
            ]
        )
        
        # 获取日志记录器
        self.logger = logging.getLogger("document_cli")
    
    def _create_storage_dirs(self):
        """创建存储目录"""
        storage_base_path = self.config_manager.get_value("storage.base_path", "./data")
        storage_output_path = self.config_manager.get_value("storage.output_path", "./output")
        
        os.makedirs(storage_base_path, exist_ok=True)
        os.makedirs(storage_output_path, exist_ok=True)
        
        self.storage_base_path = storage_base_path
        self.storage_output_path = storage_output_path
    
    def _parse_args(self):
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
            """
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
        process_parser.add_argument("--processors", help="要使用的处理器，逗号分隔，例如：OCRProcessor,StructureProcessor")
        
        # info命令
        info_parser = subparsers.add_parser("info", help="查询文档信息")
        info_parser.add_argument("--id", required=True, help="文档ID")
        
        # list命令
        list_parser = subparsers.add_parser("list", help="列出所有已处理的文档")
        list_parser.add_argument("--status", help="按状态筛选文档，例如：completed,error")
        list_parser.add_argument("--tag", help="按标签筛选文档")
        list_parser.add_argument("--format", choices=["table", "json"], default="table", 
                                 help="输出格式：表格或JSON")
        
        # export命令
        export_parser = subparsers.add_parser("export", help="导出处理后的文档")
        export_parser.add_argument("--id", required=True, help="文档ID")
        export_parser.add_argument("--format", choices=["markdown", "pdf", "text"], default="markdown", 
                                  help="导出格式")
        export_parser.add_argument("--output", help="输出文件路径")
        
        # delete命令
        delete_parser = subparsers.add_parser("delete", help="删除文档")
        delete_parser.add_argument("--id", required=True, help="文档ID")
        delete_parser.add_argument("--confirm", action="store_true", help="确认删除，不提示")
        
        # 全局选项
        parser.add_argument("--config", default=self.config_path, help="配置文件路径")
        parser.add_argument("--verbose", "-v", action="store_true", help="显示详细日志")
        
        return parser.parse_args()
    
    def run(self):
        """运行命令行界面"""
        args = self._parse_args()
        
        # 更新配置文件路径
        if args.config != self.config_path:
            self.config_path = args.config
            self.config_manager = ConfigManager(self.config_path)
        
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
    
    def _handle_upload(self, args):
        """处理上传命令"""
        self.logger.info(f"处理上传命令: {args}")
        
        # 准备要处理的文件列表
        file_paths = []
        
        if args.file:
            # 处理单个文件
            if os.path.isfile(args.file):
                file_paths.append(args.file)
                self.logger.info(f"添加文件：{args.file}")
            else:
                self.logger.error(f"文件不存在：{args.file}")
                return
        
        elif args.directory:
            # 处理目录中的文件
            if not os.path.isdir(args.directory):
                self.logger.error(f"目录不存在：{args.directory}")
                return
            
            # 确定文件扩展名过滤
            extensions = [".pdf", ".jpg", ".png", ".tiff", ".jpeg", ".bmp"]  # 默认扩展名
            if args.extensions:
                extensions = ["." + ext.lower().strip() for ext in args.extensions.split(",")]
            
            # 查找文件
            if args.recursive:
                pattern = os.path.join(args.directory, "**", "*")
                for file_path in glob.glob(pattern, recursive=True):
                    if os.path.isfile(file_path) and os.path.splitext(file_path)[1].lower() in extensions:
                        file_paths.append(file_path)
                        self.logger.debug(f"添加文件：{file_path}")
            else:
                pattern = os.path.join(args.directory, "*")
                for file_path in glob.glob(pattern):
                    if os.path.isfile(file_path) and os.path.splitext(file_path)[1].lower() in extensions:
                        file_paths.append(file_path)
                        self.logger.debug(f"添加文件：{file_path}")
            
            self.logger.info(f"从目录 {args.directory} 中找到 {len(file_paths)} 个文件")
        
        # 处理文件
        if not file_paths:
            self.logger.warning("没有找到要处理的文件")
            return
        
        # 在实际应用中，这里应调用Pipeline处理文件
        # 由于处理器尚未实现，这里模拟处理过程
        for file_path in file_paths:
            print(f"处理文件：{file_path}")
            
            # 创建Document对象
            document = Document(file_path)
            
            # 模拟存储路径
            document_dir = os.path.join(self.storage_base_path, document.document_id)
            os.makedirs(document_dir, exist_ok=True)
            
            # 模拟处理流程
            start_time = time.time()
            
            # 这里应调用Pipeline.process_document
            # 目前我们模拟一个成功的处理结果
            print(f"  文档ID：{document.document_id}")
            print(f"  状态：处理中...")
            time.sleep(1)  # 模拟处理时间
            
            # 更新状态
            document.update_status("completed")
            
            # 计算处理时间
            elapsed_time = time.time() - start_time
            
            print(f"  状态：已完成")
            print(f"  用时：{elapsed_time:.2f}秒")
            print(f"  输出目录：{document_dir}")
            
            # 将文档信息保存到JSON文件中
            document_info_path = os.path.join(document_dir, "document.json")
            with open(document_info_path, 'w', encoding='utf-8') as f:
                json.dump(document.to_dict(), f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"文件 {file_path} 处理完成，ID：{document.document_id}")
    
    def _handle_process(self, args):
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
            with open(document_info_path, 'r', encoding='utf-8') as f:
                document_data = json.load(f)
            
            # 创建Document对象
            document = Document.from_dict(document_data)
            self.logger.info(f"已加载文档：{document_id}")
            
            # 获取要使用的处理器
            if args.processors:
                processor_names = [name.strip() for name in args.processors.split(",")]
                self.logger.info(f"将使用处理器：{processor_names}")
                
                # 在实际应用中，这里应根据名称获取处理器对象
                # 并按指定顺序执行处理
            
            # 模拟处理流程
            print(f"重新处理文档：{document_id}")
            print(f"  文件名：{document.file_name}")
            print(f"  状态：处理中...")
            
            start_time = time.time()
            time.sleep(1)  # 模拟处理时间
            
            # 更新状态
            document.update_status("completed")
            
            # 计算处理时间
            elapsed_time = time.time() - start_time
            
            # 保存更新后的文档信息
            with open(document_info_path, 'w', encoding='utf-8') as f:
                json.dump(document.to_dict(), f, ensure_ascii=False, indent=2)
            
            print(f"  状态：已完成")
            print(f"  用时：{elapsed_time:.2f}秒")
            
            self.logger.info(f"文档 {document_id} 重新处理完成")
            
        except Exception as e:
            self.logger.error(f"处理文档 {document_id} 时出错: {str(e)}")
            print(f"错误：处理文档时出错：{str(e)}")
    
    def _handle_info(self, args):
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
            with open(document_info_path, 'r', encoding='utf-8') as f:
                document_data = json.load(f)
            
            # 创建Document对象
            document = Document.from_dict(document_data)
            
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
    
    def _handle_list(self, args):
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
                if os.path.isdir(document_dir) and os.path.exists(document_info_path):
                    try:
                        with open(document_info_path, 'r', encoding='utf-8') as f:
                            document_data = json.load(f)
                        
                        # 应用过滤器
                        if status_filter and document_data.get("status") not in status_filter:
                            continue
                            
                        if tag_filter and tag_filter not in document_data.get("tags", []):
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
                
                # 打印表头
                print(f"{'文档ID':<36} | {'文件名':<20} | {'状态':<10} | {'创建时间':<20} | {'标签':<20}")
                print("-" * 120)
                
                # 打印每个文档
                for doc in documents:
                    tags = ", ".join(doc.get("tags", []))[:20] if doc.get("tags") else "[]"
                    document_id = doc.get('document_id', 'N/A')
                    file_name = doc.get('file_name', 'N/A')[:20]
                    status = doc.get('status', 'N/A')
                    creation_time = doc.get('creation_time', 'N/A')[:20]
                    print(f"{document_id:<36} | {file_name:<20} | {status:<10} | {creation_time:<20} | {tags:<20}")
            
            self.logger.info(f"列出了 {len(documents)} 个文档")
            
        except Exception as e:
            self.logger.error(f"列出文档时出错: {str(e)}")
            print(f"错误：列出文档时出错：{str(e)}")
    
    def _handle_export(self, args):
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
            with open(document_info_path, 'r', encoding='utf-8') as f:
                document_data = json.load(f)
            
            # 创建Document对象
            document = Document.from_dict(document_data)
            
            # 确定输出文件路径
            output_path = args.output
            if not output_path:
                # 如果未指定输出路径，使用默认路径
                filename = f"{document.file_name.rsplit('.', 1)[0]}.{export_format}"
                output_path = os.path.join(self.storage_output_path, filename)
            
            # 检查输出目录是否存在
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # 导出文档
            print(f"导出文档：{document_id}")
            print(f"  格式：{export_format}")
            print(f"  输出路径：{output_path}")
            
            # 模拟导出过程
            # 在实际应用中，应从文档内容中获取对应格式的内容
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(f"# 文档：{document.file_name}\n\n")
                f.write(f"ID: {document.document_id}\n")
                f.write(f"创建时间: {document.creation_time}\n\n")
                f.write("这是一个示例导出文件，实际应用中应包含文档的真实内容。\n")
            
            print(f"  导出完成！")
            
            self.logger.info(f"已将文档 {document_id} 导出为 {export_format} 格式：{output_path}")
            
        except Exception as e:
            self.logger.error(f"导出文档 {document_id} 时出错: {str(e)}")
            print(f"错误：导出文档时出错：{str(e)}")
    
    def _handle_delete(self, args):
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
            if confirm.lower() != 'y':
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
    
    def _print_usage(self):
        """打印使用说明"""
        # 通过创建解析器并打印帮助来显示使用说明
        parser = argparse.ArgumentParser(
            description="文档处理命令行界面 - 提供文档上传、处理和查询功能"
        )
        subparsers = parser.add_subparsers(dest="command", help="子命令")
        
        # upload命令
        upload_parser = subparsers.add_parser("upload", help="上传并处理文档")
        
        # process命令
        process_parser = subparsers.add_parser("process", help="处理已上传的文档")
        
        # info命令
        info_parser = subparsers.add_parser("info", help="查询文档信息")
        
        # list命令
        list_parser = subparsers.add_parser("list", help="列出所有已处理的文档")
        
        # export命令
        export_parser = subparsers.add_parser("export", help="导出处理后的文档")
        
        # delete命令
        delete_parser = subparsers.add_parser("delete", help="删除文档")
        
        # 全局选项
        parser.add_argument("--config", help="配置文件路径")
        parser.add_argument("--verbose", "-v", action="store_true", help="显示详细日志")
        
        # 打印帮助信息
        parser.print_help()


def main():
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

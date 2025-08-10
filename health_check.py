#!/usr/bin/env python3
"""
Academic RAG Server Health Check Script

This script provides comprehensive health checks for the academic RAG server,
including system components, dependencies, and service availability.
"""

import sys
import os
import json
import time
import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import traceback

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("health_check")


class HealthCheckResult:
    """Health check result container"""
    
    def __init__(self, component: str, status: str, message: str, details: Optional[Dict[str, Any]] = None):
        self.component = component
        self.status = status  # "healthy", "unhealthy", "warning"
        self.message = message
        self.details = details or {}
        self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "component": self.component,
            "status": self.status,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp.isoformat()
        }
    
    def is_healthy(self) -> bool:
        return self.status == "healthy"


class HealthChecker:
    """Comprehensive health checker for the academic RAG system"""
    
    def __init__(self):
        self.results: List[HealthCheckResult] = []
    
    def add_result(self, result: HealthCheckResult) -> None:
        """Add a health check result"""
        self.results.append(result)
        logger.info(f"{result.component}: {result.status} - {result.message}")
    
    def check_python_environment(self) -> HealthCheckResult:
        """Check Python environment and version"""
        try:
            python_version = sys.version
            if sys.version_info >= (3, 8):
                return HealthCheckResult(
                    "python_environment",
                    "healthy",
                    f"Python {python_version} is supported",
                    {"version": python_version, "executable": sys.executable}
                )
            else:
                return HealthCheckResult(
                    "python_environment",
                    "unhealthy",
                    f"Python {python_version} is too old (minimum 3.8 required)",
                    {"version": python_version}
                )
        except Exception as e:
            return HealthCheckResult(
                "python_environment",
                "unhealthy",
                f"Failed to check Python environment: {str(e)}"
            )
    
    def check_required_packages(self) -> HealthCheckResult:
        """Check if required packages are installed"""
        required_packages = [
            "haystack",
            "sentence_transformers",
            "torch",
            "numpy",
            "pandas",
            "mcp"
        ]
        
        optional_packages = [
            "pymilvus",
            "opencv-python",
            "pillow",
            "psutil"
        ]
        
        missing_required = []
        missing_optional = []
        installed_versions = {}
        
        # Check required packages
        for package in required_packages:
            try:
                if package == "opencv-python":
                    import cv2
                    installed_versions[package] = cv2.__version__
                else:
                    module = __import__(package.replace("-", "_"))
                    version = getattr(module, '__version__', 'unknown')
                    installed_versions[package] = version
            except ImportError:
                missing_required.append(package)
        
        # Check optional packages
        for package in optional_packages:
            try:
                if package == "opencv-python":
                    import cv2
                    installed_versions[package] = cv2.__version__
                elif package == "pymilvus":
                    import pymilvus
                    installed_versions[package] = pymilvus.__version__
                else:
                    module = __import__(package.replace("-", "_"))
                    version = getattr(module, '__version__', 'unknown')
                    installed_versions[package] = version
            except ImportError:
                missing_optional.append(package)
        
        if missing_required:
            return HealthCheckResult(
                "required_packages",
                "unhealthy",
                f"Missing required packages: {', '.join(missing_required)}",
                {
                    "missing_required": missing_required,
                    "missing_optional": missing_optional,
                    "installed": installed_versions
                }
            )
        elif missing_optional:
            return HealthCheckResult(
                "required_packages",
                "warning",
                f"Missing optional packages: {', '.join(missing_optional)}",
                {
                    "missing_optional": missing_optional,
                    "installed": installed_versions
                }
            )
        else:
            return HealthCheckResult(
                "required_packages",
                "healthy",
                "All required and optional packages are installed",
                {"installed": installed_versions}
            )
    
    def check_configuration(self) -> HealthCheckResult:
        """Check configuration files and settings"""
        try:
            from core.config_manager import ConfigManager
            
            config_manager = ConfigManager()
            config_path = config_manager.config_file_path
            
            if not os.path.exists(config_path):
                return HealthCheckResult(
                    "configuration",
                    "unhealthy",
                    f"Configuration file not found: {config_path}"
                )
            
            # Test configuration loading
            processors_config = config_manager.get_value("processors", {})
            rag_config = config_manager.get_value("rag_settings", {})
            
            essential_configs = {
                "processors": len(processors_config),
                "rag_settings": bool(rag_config),
                "vector_db": bool(config_manager.get_value("vector_db", {})),
                "llm": bool(config_manager.get_value("llm", {}))
            }
            
            return HealthCheckResult(
                "configuration",
                "healthy",
                f"Configuration loaded successfully from {config_path}",
                {
                    "config_path": config_path,
                    "essential_configs": essential_configs
                }
            )
            
        except Exception as e:
            return HealthCheckResult(
                "configuration",
                "unhealthy",
                f"Configuration check failed: {str(e)}",
                {"error": str(e), "traceback": traceback.format_exc()}
            )
    
    def check_storage_directories(self) -> HealthCheckResult:
        """Check storage directories and permissions"""
        try:
            from core.config_manager import ConfigManager
            
            config_manager = ConfigManager()
            base_path = config_manager.get_value("storage.base_path", "./data")
            output_path = config_manager.get_value("storage.output_path", "./output")
            
            directories_to_check = [
                ("base_path", base_path),
                ("output_path", output_path),
                ("logs", "./logs"),
                ("temp", "./temp")
            ]
            
            directory_status = {}
            issues = []
            
            for name, path in directories_to_check:
                abs_path = os.path.abspath(path)
                
                if not os.path.exists(abs_path):
                    try:
                        os.makedirs(abs_path, exist_ok=True)
                        directory_status[name] = {"path": abs_path, "status": "created"}
                    except Exception as e:
                        directory_status[name] = {"path": abs_path, "status": "failed", "error": str(e)}
                        issues.append(f"Cannot create {name}: {str(e)}")
                else:
                    # Check write permissions
                    test_file = os.path.join(abs_path, ".health_check_test")
                    try:
                        with open(test_file, 'w') as f:
                            f.write("test")
                        os.remove(test_file)
                        directory_status[name] = {"path": abs_path, "status": "accessible"}
                    except Exception as e:
                        directory_status[name] = {"path": abs_path, "status": "no_write", "error": str(e)}
                        issues.append(f"No write permission for {name}: {str(e)}")
            
            if issues:
                return HealthCheckResult(
                    "storage_directories",
                    "unhealthy",
                    f"Storage directory issues: {'; '.join(issues)}",
                    {"directories": directory_status}
                )
            else:
                return HealthCheckResult(
                    "storage_directories",
                    "healthy",
                    "All storage directories are accessible",
                    {"directories": directory_status}
                )
                
        except Exception as e:
            return HealthCheckResult(
                "storage_directories",
                "unhealthy",
                f"Storage directory check failed: {str(e)}"
            )
    
    def check_milvus_connection(self) -> HealthCheckResult:
        """Check Milvus database connection"""
        try:
            from document_stores.milvus_store import MILVUS_AVAILABLE, MilvusDocumentStore
            from core.config_manager import ConfigManager
            
            if not MILVUS_AVAILABLE:
                return HealthCheckResult(
                    "milvus_connection",
                    "warning",
                    "Milvus package not available (pymilvus not installed)"
                )
            
            config_manager = ConfigManager()
            milvus_config = config_manager.get_value("vector_db.milvus", {})
            
            if not milvus_config:
                return HealthCheckResult(
                    "milvus_connection",
                    "warning",
                    "Milvus configuration not found, using in-memory storage"
                )
            
            # Test connection with a temporary collection
            test_config = milvus_config.copy()
            test_config["collection_name"] = f"health_check_{int(time.time())}"
            
            milvus_store = MilvusDocumentStore(test_config)
            
            # Get collection info to test connection
            info = milvus_store.get_collection_info()
            milvus_store.close()
            
            return HealthCheckResult(
                "milvus_connection",
                "healthy",
                f"Milvus connection successful to {milvus_config.get('host', 'localhost')}:{milvus_config.get('port', 19530)}",
                {
                    "host": milvus_config.get('host', 'localhost'),
                    "port": milvus_config.get('port', 19530),
                    "database": milvus_config.get('database', 'default'),
                    "collection_info": info
                }
            )
            
        except Exception as e:
            return HealthCheckResult(
                "milvus_connection",
                "unhealthy",
                f"Milvus connection failed: {str(e)}",
                {"error": str(e)}
            )
    
    def check_pipeline_components(self) -> HealthCheckResult:
        """Check pipeline components can be loaded"""
        try:
            from core.pipeline import Pipeline
            from mcp_server import load_processors
            
            # Test processor loading
            processors = load_processors()
            
            # Test pipeline creation
            pipeline = Pipeline("HealthCheckPipeline")
            for processor in processors:
                pipeline.add_processor(processor)
            
            processor_info = {
                "count": len(processors),
                "processors": [p.get_name() for p in processors],
                "stages": [p.get_stage() for p in processors]
            }
            
            return HealthCheckResult(
                "pipeline_components",
                "healthy",
                f"Successfully loaded {len(processors)} processors",
                processor_info
            )
            
        except Exception as e:
            return HealthCheckResult(
                "pipeline_components",
                "unhealthy",
                f"Pipeline component check failed: {str(e)}",
                {"error": str(e), "traceback": traceback.format_exc()}
            )
    
    def check_rag_pipeline(self) -> HealthCheckResult:
        """Check RAG pipeline initialization"""
        try:
            from rag.haystack_pipeline import RAGPipelineFactory
            from connectors.haystack_llm_connector import HaystackLLMConnector
            from core.config_manager import ConfigManager
            
            config_manager = ConfigManager()
            
            # Test LLM connector
            llm_config = config_manager.get_value("llm", {})
            if not llm_config:
                return HealthCheckResult(
                    "rag_pipeline",
                    "warning",
                    "LLM configuration not found, RAG functionality limited"
                )
            
            llm_connector = HaystackLLMConnector(config=llm_config)
            
            # Test RAG pipeline creation
            rag_config = config_manager.get_value("rag_settings", {})
            rag_pipeline = RAGPipelineFactory.create_pipeline(
                llm_connector=llm_connector,
                config=rag_config
            )
            
            return HealthCheckResult(
                "rag_pipeline",
                "healthy",
                "RAG pipeline initialized successfully",
                {
                    "llm_type": llm_config.get("type", "unknown"),
                    "rag_config": rag_config
                }
            )
            
        except Exception as e:
            return HealthCheckResult(
                "rag_pipeline",
                "unhealthy",
                f"RAG pipeline check failed: {str(e)}",
                {"error": str(e)}
            )
    
    def check_system_resources(self) -> HealthCheckResult:
        """Check system resources (memory, disk space, etc.)"""
        try:
            import psutil
            
            # Memory check
            memory = psutil.virtual_memory()
            memory_gb = memory.total / (1024**3)
            memory_available_gb = memory.available / (1024**3)
            memory_percent = memory.percent
            
            # Disk space check
            disk = psutil.disk_usage('.')
            disk_total_gb = disk.total / (1024**3)
            disk_free_gb = disk.free / (1024**3)
            disk_percent = (disk.used / disk.total) * 100
            
            # CPU check
            cpu_count = psutil.cpu_count()
            cpu_percent = psutil.cpu_percent(interval=1)
            
            resources = {
                "memory": {
                    "total_gb": round(memory_gb, 2),
                    "available_gb": round(memory_available_gb, 2),
                    "used_percent": memory_percent
                },
                "disk": {
                    "total_gb": round(disk_total_gb, 2),
                    "free_gb": round(disk_free_gb, 2),
                    "used_percent": round(disk_percent, 2)
                },
                "cpu": {
                    "count": cpu_count,
                    "usage_percent": cpu_percent
                }
            }
            
            # Determine status based on resource usage
            issues = []
            if memory_percent > 90:
                issues.append(f"High memory usage: {memory_percent}%")
            if disk_percent > 90:
                issues.append(f"Low disk space: {disk_percent}% used")
            if cpu_percent > 90:
                issues.append(f"High CPU usage: {cpu_percent}%")
            
            if issues:
                return HealthCheckResult(
                    "system_resources",
                    "warning",
                    f"Resource concerns: {'; '.join(issues)}",
                    resources
                )
            else:
                return HealthCheckResult(
                    "system_resources",
                    "healthy",
                    "System resources are adequate",
                    resources
                )
                
        except ImportError:
            return HealthCheckResult(
                "system_resources",
                "warning",
                "psutil not available, cannot check system resources"
            )
        except Exception as e:
            return HealthCheckResult(
                "system_resources",
                "unhealthy",
                f"System resource check failed: {str(e)}"
            )
    
    async def check_async_functionality(self) -> HealthCheckResult:
        """Check async functionality"""
        try:
            from core.pipeline import Pipeline
            from models.document import Document
            
            # Create a simple test document
            test_doc = Document("test_health_check.txt")
            test_doc.store_content("test", "Health check test content")
            
            # Test async pipeline
            pipeline = Pipeline("AsyncHealthCheck")
            
            # Simple async test
            result = await pipeline.process_document(test_doc)
            
            return HealthCheckResult(
                "async_functionality",
                "healthy",
                "Async functionality working correctly",
                {"test_result": result.is_successful()}
            )
            
        except Exception as e:
            return HealthCheckResult(
                "async_functionality",
                "unhealthy",
                f"Async functionality check failed: {str(e)}",
                {"error": str(e)}
            )
    
    async def run_all_checks(self) -> Dict[str, Any]:
        """Run all health checks"""
        logger.info("Starting comprehensive health check...")
        
        # Synchronous checks
        checks = [
            self.check_python_environment,
            self.check_required_packages,
            self.check_configuration,
            self.check_storage_directories,
            self.check_milvus_connection,
            self.check_pipeline_components,
            self.check_rag_pipeline,
            self.check_system_resources
        ]
        
        for check_func in checks:
            result = check_func()
            self.add_result(result)
        
        # Asynchronous checks
        async_result = await self.check_async_functionality()
        self.add_result(async_result)
        
        # Compile overall status
        healthy_count = sum(1 for r in self.results if r.status == "healthy")
        warning_count = sum(1 for r in self.results if r.status == "warning")
        unhealthy_count = sum(1 for r in self.results if r.status == "unhealthy")
        
        overall_status = "healthy"
        if unhealthy_count > 0:
            overall_status = "unhealthy"
        elif warning_count > 0:
            overall_status = "warning"
        
        summary = {
            "overall_status": overall_status,
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_checks": len(self.results),
                "healthy": healthy_count,
                "warnings": warning_count,
                "unhealthy": unhealthy_count
            },
            "checks": [result.to_dict() for result in self.results]
        }
        
        logger.info(f"Health check completed - Status: {overall_status}")
        logger.info(f"Results: {healthy_count} healthy, {warning_count} warnings, {unhealthy_count} unhealthy")
        
        return summary


def check_system_health() -> bool:
    """
    Simple health check function for Docker healthcheck
    Returns True if system is healthy, False otherwise
    """
    try:
        checker = HealthChecker()
        
        # Run essential checks only for quick health check
        essential_results = [
            checker.check_python_environment(),
            checker.check_configuration(),
            checker.check_storage_directories()
        ]
        
        # Check if any essential component is unhealthy
        for result in essential_results:
            if result.status == "unhealthy":
                return False
        
        return True
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return False


async def main():
    """Main function for running comprehensive health check"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Academic RAG Server Health Check")
    parser.add_argument("--output", "-o", help="Output file for health check results")
    parser.add_argument("--format", "-f", choices=["json", "text"], default="text", help="Output format")
    parser.add_argument("--simple", "-s", action="store_true", help="Run simple health check (for Docker)")
    
    args = parser.parse_args()
    
    if args.simple:
        # Simple health check for Docker
        is_healthy = check_system_health()
        if args.format == "json":
            result = {"healthy": is_healthy, "timestamp": datetime.now().isoformat()}
            print(json.dumps(result, indent=2))
        else:
            print(f"Health Status: {'HEALTHY' if is_healthy else 'UNHEALTHY'}")
        
        sys.exit(0 if is_healthy else 1)
    
    # Comprehensive health check
    checker = HealthChecker()
    results = await checker.run_all_checks()
    
    # Output results
    if args.format == "json":
        output = json.dumps(results, indent=2)
    else:
        # Text format
        lines = [
            "=" * 80,
            "ACADEMIC RAG SERVER HEALTH CHECK REPORT",
            "=" * 80,
            f"Overall Status: {results['overall_status'].upper()}",
            f"Timestamp: {results['timestamp']}",
            f"Total Checks: {results['summary']['total_checks']}",
            f"Healthy: {results['summary']['healthy']}",
            f"Warnings: {results['summary']['warnings']}",
            f"Unhealthy: {results['summary']['unhealthy']}",
            "",
            "DETAILED RESULTS:",
            "-" * 80
        ]
        
        for check in results['checks']:
            status_symbol = {
                'healthy': '✓',
                'warning': '⚠',
                'unhealthy': '✗'
            }.get(check['status'], '?')
            
            lines.extend([
                f"{status_symbol} {check['component'].upper()}: {check['status'].upper()}",
                f"  Message: {check['message']}",
                ""
            ])
        
        output = "\\n".join(lines)
    
    # Write to file or stdout
    if args.output:
        with open(args.output, 'w') as f:
            f.write(output)
        print(f"Health check results written to {args.output}")
    else:
        print(output)
    
    # Exit with appropriate code
    sys.exit(0 if results['overall_status'] != 'unhealthy' else 1)


if __name__ == "__main__":
    asyncio.run(main())
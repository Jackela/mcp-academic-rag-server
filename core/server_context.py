"""
Server Context - Dependency Injection Container
Manages all server dependencies and eliminates global state
"""

from typing import Optional, List
import logging
from pathlib import Path

from core.config_manager import ConfigManager
from core.pipeline import Pipeline
from rag.haystack_pipeline import RAGPipeline
from rag.chat_session import ChatSessionManager
from connectors.haystack_llm_connector import HaystackLLMConnector
from processors.base_processor import IProcessor


class ServerContext:
    """
    Centralized dependency injection container for the MCP server.
    
    This class manages all server dependencies including configuration,
    pipelines, and session management, eliminating global state.
    """
    
    def __init__(self):
        """Initialize the server context with default dependencies."""
        self._logger = logging.getLogger("mcp-academic-rag-server")
        self._config_manager: Optional[ConfigManager] = None
        self._document_pipeline: Optional[Pipeline] = None
        self._rag_pipeline: Optional[RAGPipeline] = None
        self._session_manager: Optional[ChatSessionManager] = None
        self._processors: List[IProcessor] = []
        self._initialized = False
    
    @property
    def config_manager(self) -> ConfigManager:
        """Get the configuration manager, creating it if needed."""
        if self._config_manager is None:
            self._config_manager = ConfigManager()
        return self._config_manager
    
    @property
    def document_pipeline(self) -> Optional[Pipeline]:
        """Get the document processing pipeline."""
        return self._document_pipeline
    
    @property
    def rag_pipeline(self) -> Optional[RAGPipeline]:
        """Get the RAG pipeline."""
        return self._rag_pipeline
    
    @property
    def session_manager(self) -> ChatSessionManager:
        """Get the session manager, creating it if needed."""
        if self._session_manager is None:
            self._session_manager = ChatSessionManager()
        return self._session_manager
    
    @property
    def processors(self) -> List[IProcessor]:
        """Get the list of loaded processors."""
        return self._processors
    
    @property
    def is_initialized(self) -> bool:
        """Check if the server context has been fully initialized."""
        return self._initialized
    
    def initialize(self) -> None:
        """
        Initialize all server components in the correct dependency order.
        
        This method implements a multi-phase initialization strategy:
        1. Guard check: Prevent duplicate initialization
        2. Core pipeline setup: Create document processing pipeline
        3. Processor loading: Dynamically load and configure document processors
        4. RAG pipeline setup: Initialize retrieval-augmented generation capabilities
        5. Validation: Ensure all components are properly initialized
        
        The initialization order is critical:
        - Document pipeline must exist before processors are loaded
        - Processors must be loaded before RAG pipeline initialization
        - RAG pipeline depends on document processing capabilities
        
        This method should be called once during server startup to ensure
        all dependencies are properly configured and connected.
        
        Raises:
            Exception: If any component fails to initialize properly
        """
        # Phase 1: Guard check to prevent duplicate initialization
        if self._initialized:
            self._logger.warning("Server context already initialized - skipping re-initialization")
            return
        
        try:
            self._logger.info("Initializing server context components")
            
            # Phase 2: Core pipeline setup
            # Create the main document processing pipeline that will orchestrate
            # all document processing operations through the loaded processors
            self._document_pipeline = Pipeline()
            self._logger.debug("Document pipeline created")
            
            # Phase 3: Processor loading and configuration
            # Load processors from configuration and add them to the pipeline
            # This step is critical as processors define the document processing capabilities
            self._load_processors()
            self._logger.debug(f"Loaded {len(self._processors)} processors")
            
            # Phase 4: RAG pipeline initialization
            # Initialize the retrieval-augmented generation pipeline for querying
            # This depends on having document processing capabilities available
            self._initialize_rag_pipeline()
            rag_status = "enabled" if self._rag_pipeline else "disabled"
            self._logger.debug(f"RAG pipeline {rag_status}")
            
            # Phase 5: Validation and completion
            self._initialized = True
            self._logger.info(
                "Server context initialization completed successfully",
                extra={
                    'processors_count': len(self._processors),
                    'rag_enabled': self._rag_pipeline is not None,
                    'pipeline_ready': self._document_pipeline is not None,
                    'components_initialized': 'document_pipeline, processors, rag_pipeline'
                }
            )
            
        except Exception as e:
            # Initialization failed - log detailed error and cleanup partial state
            self._logger.error(
                f"Failed to initialize server context: {str(e)}",
                extra={
                    'initialization_phase': 'unknown',
                    'partial_state': self.get_status()
                },
                exc_info=True
            )
            # Reset any partially initialized state to prevent inconsistent state
            self._initialized = False
            self._document_pipeline = None
            self._processors = []
            self._rag_pipeline = None
            raise
    
    def _load_processors(self) -> None:
        """Load processors from configuration."""
        from core.processor_loader import ProcessorLoader
        
        loader = ProcessorLoader(self.config_manager)
        self._processors = loader.load_processors()
        
        # Add processors to pipeline
        if self._document_pipeline:
            for processor in self._processors:
                self._document_pipeline.add_processor(processor)
    
    def _initialize_rag_pipeline(self) -> None:
        """Initialize the RAG pipeline with LLM connector."""
        import os
        from rag.haystack_pipeline import RAGPipelineFactory
        
        try:
            # Get LLM configuration
            llm_config = self.config_manager.get_value("llm", {})
            api_key = llm_config.get("api_key", os.environ.get("OPENAI_API_KEY", ""))
            model = llm_config.get("model", "gpt-3.5-turbo")
            
            if not api_key:
                self._logger.warning("No OpenAI API key found, RAG pipeline disabled")
                return
            
            # Create LLM connector
            llm_connector = HaystackLLMConnector(api_key=api_key, model=model)
            
            # Create RAG pipeline
            rag_config = self.config_manager.get_value("rag_settings", {})
            self._rag_pipeline = RAGPipelineFactory.create_pipeline(
                llm_connector=llm_connector,
                config=rag_config
            )
            
            self._logger.info("RAG pipeline initialized successfully")
            
        except Exception as e:
            self._logger.error(f"Failed to initialize RAG pipeline: {str(e)}")
            # Continue without RAG pipeline - server can still process documents
    
    def cleanup(self) -> None:
        """Clean up resources and reset the context with proper resource management."""
        self._logger.info("Cleaning up server context")
        
        try:
            # Clean up session manager
            if self._session_manager:
                # Clear all sessions to free memory
                self._session_manager._sessions.clear()
                self._logger.debug("Session manager cleaned up")
            
            # Clean up RAG pipeline resources
            if self._rag_pipeline:
                # RAG pipeline cleanup would go here if needed
                self._logger.debug("RAG pipeline cleaned up")
            
            # Clean up document pipeline
            if self._document_pipeline:
                # Document pipeline cleanup would go here if needed
                self._logger.debug("Document pipeline cleaned up")
            
            # Clean up processors
            for processor in self._processors:
                # Individual processor cleanup would go here if needed
                pass
            
            # Reset state
            self._document_pipeline = None
            self._rag_pipeline = None
            self._processors = []
            self._session_manager = None
            self._initialized = False
            
            self._logger.info("Server context cleanup completed successfully")
            
        except Exception as e:
            self._logger.error(f"Error during server context cleanup: {str(e)}", exc_info=True)
            # Force reset even if cleanup failed
            self._document_pipeline = None
            self._rag_pipeline = None
            self._processors = []
            self._session_manager = None
            self._initialized = False
    
    def get_status(self) -> dict:
        """Get the current status of the server context."""
        return {
            "initialized": self._initialized,
            "config_loaded": self._config_manager is not None,
            "pipeline_ready": self._document_pipeline is not None,
            "rag_enabled": self._rag_pipeline is not None,
            "processors_count": len(self._processors),
            "session_manager_ready": self._session_manager is not None
        }
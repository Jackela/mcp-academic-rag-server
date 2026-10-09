MCP Academic RAG Server Documentation
=====================================

Welcome to the comprehensive documentation for the MCP Academic RAG Server - a system for academic document processing and retrieval-augmented generation.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started
   
   quickstart-guide
   user-guide

.. toctree::
   :maxdepth: 2
   :caption: User Documentation
   
   mcp-server-usage-guide
   multi-model-setup-guide
   workflow-command-design

.. toctree::
   :maxdepth: 2
   :caption: Developer Documentation
   
   developer-guide
   developer-guide-enhanced
   api-documentation-system
   docstring-standards

.. toctree::
   :maxdepth: 2
   :caption: Architecture & Design
   
   architecture-overview
   vector-storage-implementation

.. toctree::
   :maxdepth: 3
   :caption: API Reference
   
   api/core
   api/connectors
   api/document_stores
   api/rag
   api/servers

.. toctree::
   :maxdepth: 1
   :caption: Maintenance and supplementary references
   :glob:

   maintenance
   api-reference
   api-rag-pipeline
   user-guide/mcp-tools-reference
   archived/*

Project Overview
================

The MCP Academic RAG Server is an enterprise-grade system that combines:

* **Document Processing**: Advanced OCR, text extraction, and preprocessing
* **Vector Storage**: High-performance vector databases for semantic search  
* **RAG Pipeline**: Retrieval-augmented generation with multiple LLM providers
* **MCP Protocol**: Native integration with AI assistants like Claude
* **Monitoring**: Comprehensive observability and performance tracking
* **Configuration**: Enterprise-level configuration management

Key Features
============

📄 **Multi-Format Support**
   Process PDF, DOCX, TXT, and other academic document formats

🔍 **Advanced Search**
   Semantic vector search with metadata filtering and hybrid retrieval

🤖 **LLM Integration**
   Support for OpenAI, Anthropic, Google, and custom LLM providers

⚡ **High Performance**
   Optimized async processing with caching and batch operations

🔧 **Enterprise Ready**
   Configuration management, monitoring, alerting, and security

🧩 **Extensible**
   Plugin architecture for custom processors and integrations

Quick Start
===========

1. **Installation**::

    pip install mcp-academic-rag-server

2. **Configuration**::

    cp config/config.json.example config/config.json
    # Edit configuration with your API keys

3. **Start Server**::

    mcp-academic-rag-server --help
    mcp-academic-rag-server

4. **Discover and call tools**:

   Connect an MCP client over stdio. See :doc:`maintenance` for the isolated
   client handshake and :doc:`user-guide/mcp-tools-reference` for tool arguments.

Architecture Highlights
=======================

The system implements a **layered, modular architecture** designed for:

* **Scalability**: Horizontal scaling with load balancing
* **Reliability**: Comprehensive error handling and monitoring
* **Maintainability**: Clean interfaces and separation of concerns
* **Performance**: Async processing and intelligent caching
* **Security**: Authentication, authorization, and data protection

Core Components:

See :doc:`architecture-overview` for the existing architecture diagrams.

Support and Community
=====================

* **Documentation**: Complete API reference and guides
* **GitHub**: `Source code and issues <https://github.com/Jackela/mcp-academic-rag-server>`_
* **Community**: Developer forums and discussions
* **Enterprise**: Professional support and consulting available

API Reference
=============

.. autosummary::
   :toctree: api
   :recursive:
   
   core
   connectors
   document_stores
   rag
   servers

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
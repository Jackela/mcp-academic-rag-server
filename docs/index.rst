MCP Academic RAG Server Documentation
=====================================

Welcome to the comprehensive documentation for the MCP Academic RAG Server - a production-ready system for academic document processing and retrieval-augmented generation.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started
   
   quickstart-guide
   user-guide
   installation

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
   
   api/modules

.. toctree::
   :maxdepth: 1
   :caption: Configuration
   
   configuration

.. toctree::
   :maxdepth: 1
   :caption: Deployment
   
   deployment

.. toctree::
   :maxdepth: 1
   :caption: Monitoring
   
   monitoring

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

    cp config/config.example.json config/config.json
    # Edit configuration with your API keys

3. **Start Server**::

    python -m mcp_rag_server

4. **Process Documents**::

    curl -X POST "http://localhost:8080/api/v1/documents" \
         -F "file=@paper.pdf" \
         -F "collection_id=research"

5. **Query Documents**::

    curl -X POST "http://localhost:8080/api/v1/query" \
         -H "Content-Type: application/json" \
         -d '{"query": "What are the main findings?", "collection_id": "research"}'

Architecture Highlights
=======================

The system implements a **layered, modular architecture** designed for:

* **Scalability**: Horizontal scaling with load balancing
* **Reliability**: Comprehensive error handling and monitoring
* **Maintainability**: Clean interfaces and separation of concerns
* **Performance**: Async processing and intelligent caching
* **Security**: Authentication, authorization, and data protection

Core Components:

.. image:: _static/architecture-diagram.png
   :alt: System Architecture
   :align: center

Support and Community
=====================

* **Documentation**: Complete API reference and guides
* **GitHub**: `Source code and issues <https://github.com/mcp/academic-rag-server>`_
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
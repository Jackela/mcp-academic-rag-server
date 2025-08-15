# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
from pathlib import Path

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'MCP Academic RAG Server'
copyright = '2024, MCP Academic RAG Server Team'
author = 'MCP Academic RAG Server Team'
version = '1.0.0'
release = '1.0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

extensions = [
    'sphinx.ext.autodoc',           # Automatic documentation from docstrings
    'sphinx.ext.autosummary',       # Generate summary tables
    'sphinx.ext.viewcode',          # Add source code links
    'sphinx.ext.napoleon',          # Google/NumPy style docstrings
    'sphinx.ext.intersphinx',       # Link to other documentation
    'sphinx.ext.todo',              # TODO items
    'sphinx.ext.coverage',          # Documentation coverage
    'sphinx.ext.mathjax',           # Mathematical notation
    'sphinx.ext.githubpages',       # GitHub Pages support
    'myst_parser',                  # Markdown support
    'sphinx_rtd_theme',             # Read the Docs theme
    'sphinxcontrib.openapi',        # OpenAPI documentation
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# Source file suffixes
source_suffix = {
    '.rst': None,
    '.md': 'myst_parser',
}

# Master document
master_doc = 'index'

# Language
language = 'en'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# Theme options
html_theme_options = {
    'canonical_url': '',
    'analytics_id': '',
    'logo_only': False,
    'display_version': True,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': False,
    'vcs_pageview_mode': '',
    'style_nav_header_background': '#2980B9',
    'collapse_navigation': False,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False
}

# Custom CSS
html_css_files = [
    'custom.css',
]

# HTML context
html_context = {
    'display_github': True,
    'github_user': 'mcp',
    'github_repo': 'academic-rag-server',
    'github_version': 'main',
    'conf_py_path': '/docs/',
}

# -- Extension configuration -------------------------------------------------

# AutoDoc configuration
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__',
    'show-inheritance': True,
}

autodoc_typehints = 'description'
autodoc_typehints_description_target = 'documented'

# Napoleon configuration for Google-style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# Intersphinx mapping
intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
    'fastapi': ('https://fastapi.tiangolo.com/', None),
    'pydantic': ('https://docs.pydantic.dev/', None),
}

# MyST Parser configuration
myst_enable_extensions = [
    "deflist",
    "tasklist",
    "substitution",
    "colon_fence",
    "linkify",
]

# TODO configuration
todo_include_todos = True

# Coverage configuration
coverage_show_missing_items = True

# -- Custom configuration for API documentation ------------------------------

# Automatically generate stub pages for API documentation
autosummary_generate = True
autosummary_generate_overwrite = True

# Mock imports for modules that might not be available during doc build
autodoc_mock_imports = [
    'faiss',
    'pymupdf',
    'pytesseract',
    'easyocr',
    'torch',
    'transformers',
    'openai',
    'anthropic',
    'google',
    'psutil',
    'redis',
    'watchdog',
]

# Custom roles and directives
def setup(app):
    """Custom Sphinx setup."""
    app.add_css_file('custom.css')
    
    # Add custom roles
    from docutils.parsers.rst import roles
    
    def api_endpoint_role(name, rawtext, text, lineno, inliner, options={}, content=[]):
        """Custom role for API endpoints."""
        from docutils import nodes
        
        node = nodes.literal(rawtext, text, **options)
        node['classes'].append('api-endpoint')
        return [node], []
    
    def config_key_role(name, rawtext, text, lineno, inliner, options={}, content=[]):
        """Custom role for configuration keys."""
        from docutils import nodes
        
        node = nodes.literal(rawtext, text, **options)
        node['classes'].append('config-key')
        return [node], []
    
    roles.register_local_role('api', api_endpoint_role)
    roles.register_local_role('config', config_key_role)

# -- LaTeX output configuration ----------------------------------------------

latex_elements = {
    'papersize': 'letterpaper',
    'pointsize': '10pt',
    'preamble': r'''
\usepackage{charter}
\usepackage[defaultsans]{lato}
\usepackage{inconsolata}
''',
    'fncychap': '\\usepackage[Bjornstrup]{fncychap}',
    'printindex': '\\footnotesize\\raggedright\\printindex',
}

latex_documents = [
    (master_doc, 'MCPAcademicRAGServer.tex', 
     'MCP Academic RAG Server Documentation', 
     'MCP Academic RAG Server Team', 'manual'),
]

# -- EPUB output configuration -----------------------------------------------

epub_title = project
epub_author = author
epub_publisher = author
epub_copyright = copyright
epub_exclude_files = ['search.html']

# -- Manual page output configuration ----------------------------------------

man_pages = [
    (master_doc, 'mcpacademicragserver', 
     'MCP Academic RAG Server Documentation',
     [author], 1)
]

# -- Texinfo output configuration --------------------------------------------

texinfo_documents = [
    (master_doc, 'MCPAcademicRAGServer', 
     'MCP Academic RAG Server Documentation',
     author, 'MCPAcademicRAGServer', 
     'Academic document processing and RAG server.',
     'Miscellaneous'),
]
# Makefile for Sphinx documentation

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD  ?= sphinx-build
SOURCEDIR    = .
BUILDDIR     = _build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Custom targets for documentation generation
clean:
	rm -rf $(BUILDDIR)/*

livehtml:
	sphinx-autobuild -b html $(SOURCEDIR) $(BUILDDIR)/html

api-docs:
	sphinx-apidoc -o api ../core ../connectors ../document_stores ../rag ../servers --force

coverage:
	sphinx-build -b coverage $(SOURCEDIR) $(BUILDDIR)/coverage

linkcheck:
	sphinx-build -b linkcheck $(SOURCEDIR) $(BUILDDIR)/linkcheck

doctest:
	sphinx-build -b doctest $(SOURCEDIR) $(BUILDDIR)/doctest

pdf:
	sphinx-build -b latex $(SOURCEDIR) $(BUILDDIR)/latex
	$(MAKE) -C $(BUILDDIR)/latex all-pdf

epub:
	sphinx-build -b epub $(SOURCEDIR) $(BUILDDIR)/epub

# Development targets
dev-setup:
	pip install -r requirements-docs.txt

dev-build: clean api-docs html

dev-serve: livehtml

# CI/CD targets
ci-build: clean api-docs html coverage linkcheck

# Deployment targets
deploy-docs: clean api-docs html
	@echo "Building documentation for deployment..."
	@if [ -d "$(BUILDDIR)/html" ]; then \
		echo "Documentation built successfully in $(BUILDDIR)/html"; \
	else \
		echo "Documentation build failed"; \
		exit 1; \
	fi

# Quality checks
quality-check: coverage linkcheck
	@echo "Running documentation quality checks..."
	@if [ -f "$(BUILDDIR)/coverage/python.txt" ]; then \
		echo "Coverage report generated"; \
		cat $(BUILDDIR)/coverage/python.txt; \
	fi
	@if [ -f "$(BUILDDIR)/linkcheck/output.txt" ]; then \
		echo "Link check completed"; \
		grep -E "(broken|redirect)" $(BUILDDIR)/linkcheck/output.txt || echo "No broken links found"; \
	fi
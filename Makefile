PYTHON ?= python3

.PHONY: help bump-version git-release

help:
	@echo "Targets:"
	@echo "  bump-version [PART=minor|major]     - bump version (defaults to patch)"
	@echo "  git-release                         - create git tag v<version> and push"
	@echo "  pypi                                - build sdist/wheel and upload via twine"
	@echo "  test-pypi                           - build and upload to TestPyPI via twine"

bump-version:
	@if [ -n "$(PART)" ]; then \
		PART_LABEL=$(PART); \
		$(PYTHON) scripts/bump_version.py $${PART_LABEL}; \
	else \
		PART_LABEL=patch; \
		$(PYTHON) scripts/bump_version.py; \
	fi; \
	echo "Version bumped ($$PART_LABEL)."

git-release:
	@VERSION=$$(sed -n "s/^__version__ = ['\"]\(.*\)['\"]/\1/p" mlx_genkit/__init__.py); \
	if [ -z "$$VERSION" ]; then echo "Could not read version"; exit 1; fi; \
	echo "Tagging v$$VERSION"; \
	git tag v$$VERSION && git push origin v$$VERSION && git push

pypi:
	$(PYTHON) -m pip install --upgrade build twine
	rm -rf dist/ build/
	$(PYTHON) -m build
	$(PYTHON) -m twine upload dist/*

test-pypi:
	$(PYTHON) -m pip install --upgrade build twine
	rm -rf dist/ build/
	$(PYTHON) -m build
	$(PYTHON) -m twine upload --repository testpypi dist/*

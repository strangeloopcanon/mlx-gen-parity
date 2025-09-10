PYTHON ?= python3

PART ?= patch

.PHONY: help bump-version git-release

help:
	@echo "Targets:"
	@echo "  bump-version PART=patch|minor|major  - bump version in pyproject + __init__"
	@echo "  git-release                         - create git tag v<version> and push"
	@echo "  pypi                                - build sdist/wheel and upload via twine"
	@echo "  test-pypi                           - build and upload to TestPyPI via twine"

bump-version:
	$(PYTHON) scripts/bump_version.py $(PART)
	@echo "Version bumped ($${PART:-patch})."

git-release:
	@VERSION=$$($(PYTHON) - <<'PY'
import re, sys
v=None
with open('mlx_gen_parity/__init__.py') as f:
    m=re.search(r"__version__\s*=\s*['\"]([^'\"]+)['\"]", f.read())
    v=m.group(1) if m else None
print(v or "")
PY
); \
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

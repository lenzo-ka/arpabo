.PHONY: venv
venv:
	python3 -m venv venv
	./venv/bin/pip install --upgrade pip
	./venv/bin/pip install -e .
	./venv/bin/pip install -r requirements-dev.txt
	./venv/bin/pre-commit install
	@echo ""
	@echo "✓ Virtual environment created!"
	@echo "✓ Pre-commit hooks installed!"
	@echo "Activate it with: source venv/bin/activate"

.PHONY: install
install:
	python -m pip install -e .

.PHONY: install-dev
install-dev:
	python -m pip install -e .
	python -m pip install -r requirements-dev.txt

.PHONY: test
test:
	pytest

.PHONY: test-fast
test-fast:
	pytest -m "not slow"

.PHONY: test-verbose
test-verbose:
	pytest -vv

.PHONY: test-asr
test-asr:
	python tests/test_asr_decode.py

.PHONY: lint
lint:
	ruff check src/
	mypy src/

.PHONY: format
format:
	ruff format src/

.PHONY: pre-commit
pre-commit:
	pre-commit run --all-files

.PHONY: build
build:
	python -m pip install --upgrade build
	python -m build

.PHONY: dist
dist: build

.PHONY: clean
clean:
	rm -rf build dist src/*.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete

.PHONY: distclean
distclean: clean
	rm -rf venv
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -rf .coverage .coverage.*
	rm -rf .ruff_cache
	rm -rf .mypy_cache
	find . -type f -name '*.arpa' -delete
	find . -type f -name '*.lm.bin' -delete
	find . -type f -name '*.fst' -delete

.PHONY: test-install
test-install: build
	python -m pip install --upgrade pip
	python -m pip install dist/*.whl

.PHONY: upload-test
upload-test: build
	python -m pip install --upgrade twine
	python -m twine upload --repository testpypi dist/*

.PHONY: upload
upload: build
	python -m pip install --upgrade twine
	python -m twine upload dist/*

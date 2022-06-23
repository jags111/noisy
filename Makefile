PYTHON=./.venv/bin/python3
PIP_FREEZE=.requirements.freeze.txt
PY_FILES=*.py noisy/
.PHONY: ci py-deps type-check lint clean

ci: $(PY_FILES) py-deps type-check lint test

type-check: $(PY_FILES)
	$(PYTHON) -m mypy --install-types --strict $(PY_FILES)

lint: $(PY_FILES)
	$(PYTHON) -m flake8 $(PY_FILES)

test: $(PY_FILES)
	$(PYTHON) -m pytest

py-deps: $(PIP_FREEZE)

$(PIP_FREEZE): requirements.txt
	$(PYTHON) -m pip install --upgrade pip && \
	$(PYTHON) -m pip install --upgrade -r requirements.txt && \
	$(PYTHON) -m pip freeze > $(PIP_FREEZE)

clean:
	rm -rf .mypy_cache/ $(PIP_FREEZE) wandb/

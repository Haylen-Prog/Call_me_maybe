VENV = .venv
VENV_PATH = ${VENV}/bin/activate
OUTPUT = output
PYTHON = python3
LLM = llm_sdk

MYPYSTRICT = mypy --exclude=${LLM} --exclude=${VENV} --strict .
FLAKE8 = flake8 --exclude=${LLM},${VENV} .
MYPYFLAGS = mypy --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs --exclude=${LLM} --exclude=${VENV} .

.PHONY: install run all debug test demo clean fclean lint lint-strict re push

install:
	@cd /goinfre/$${USER}/containers && uv venv
	@rm -rf .venv; \
	ln -s /goinfre/$${USER}/containers/.venv .venv; \
	uv sync; \
	uv add --editable ./llm_sdk; \
	uv add accelerate mypy flake8 numpy pydantic

run:
	@test -L .venv || (echo "[ERROR] Run make install first"; exit 1)
	@echo "Python executable:"; \
	uv run python -c "import sys; print(sys.executable)"; \
	echo ""; \
	echo "Prefix:"; \
	uv run python -c "import sys; print(sys.prefix)"; \
	echo ""; \
	echo "Base prefix:"; \
	uv run python -c "import sys; print(sys.base_prefix)"; \
	echo "" ; \
	echo "Inside virtualenv:"; \
	uv run python -c "import sys; print(sys.prefix != sys.base_prefix)"; \
	echo "" ; \
	uv run python -m src

all: install run

debug:
	@test -L .venv || (echo "[ERROR] Run make install first"; exit 1)
	@echo "Python executable:"; \
	uv run python -c "import sys; print(sys.executable)"; \
	echo ""; \
	echo "Prefix:"; \
	uv run python -c "import sys; print(sys.prefix)"; \
	echo ""; \
	echo "Base prefix:"; \
	uv run python -c "import sys; print(sys.base_prefix)"; \
	echo "" ; \
	echo "Inside virtualenv:"; \
	uv run python -c "import sys; print(sys.prefix != sys.base_prefix)"; \
	echo "" ; \
	uv run python -m src --debug

clean:
	@find . -type d -name "__pycache__" -exec rm -rf {} +; rm -rf .mypy_cache

fclean: clean
	@rm -rf ${VENV} ${OUTPUT}; \
	cd /goinfre/$${USER}/containers; \
	rm -rf ${VENV}

lint: install
	@. ${VENV_PATH}; \
	${FLAKE8} && ${MYPYFLAGS}

lint-strict: install
	@. ${VENV_PATH}; \
	${FLAKE8} && ${MYPYSTRICT}

re: fclean all

push:
	@read -p "Commit message: " MESSAGE; \
	git add .; \
	git status; \
	git commit -m "$$MESSAGE"; \
	git push

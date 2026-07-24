VENV = .venv
VENV_PATH = ${VENV}/bin/activate
PYTHON = python3
MYPYSTRICT = mypy --exclude=${VENV} --strict .
FLAKE8 = flake8 --exclude=${VENV} .
MYPYFLAGS = mypy --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs --exclude=${VENV} .

.PHONY: install run all debug clean fclean lint lint-strict push

install:
	@cd /goinfre/$${USER}/containers; \
	uv venv
	@rm -rf .venv; \
	ln -s /goinfre/$${USER}/containers/.venv .venv; \
	uv sync; \
	uv add --editable ./llm_sdk

run:
	@echo "Python executable:"; \
	uv run python -c "import sys; print(sys.executable)"; \
	echo ""; \
	echo "Prefix:"; \
	uv run python -c "import sys; print(sys.prefix)"; \
	echo ""; \
	echo "Base prefix:"; \
	uv run python -c "import sys; print(sys.base_prefix)"; \
	echo "" ; \
	sleep 1; \
	echo "Inside virtualenv:"; \
	uv run python -c "import sys; print(sys.prefix != sys.base_prefix)"; \
	echo "" ; \
	sleep 2; \
	uv run python -m call_me_maybe

all: install run

debug:
	@pdb ${NAME}

clean:
	@find . -type d -name "__pycache__" -exec rm -rf {} +; rm -rf .mypy_cache

fclean: clean
	@rm -rf ${VENV}; \
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
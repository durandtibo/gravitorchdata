SHELL=/bin/bash
NAME=gravitorchdata
SOURCE=src/$(NAME)
TESTS=tests
UNIT_TESTS=tests/unit
INTEGRATION_TESTS=tests/integration

.PHONY : conda
conda :
	conda env create -f environment.yaml --force

.PHONY : config-poetry
config-poetry :
	poetry config experimental.system-git-client true
	poetry config --list

.PHONY : install
install :
	poetry install --no-interaction --all-extras
	pip install --upgrade "torch>=2.1.1"  # TODO: https://github.com/pytorch/pytorch/issues/100974

.PHONY : install-all
install-all :
	poetry install --no-interaction --with exp --all-extras
	pip install --upgrade "torch>=2.1.1"  # TODO: https://github.com/pytorch/pytorch/issues/100974

.PHONY : install-min
install-min :
	poetry install --no-interaction
	pip install --upgrade "torch>=2.1.1"  # TODO: https://github.com/pytorch/pytorch/issues/100974

.PHONY : update
update :
	-poetry self update
	poetry update
	-pre-commit autoupdate

.PHONY : lint
lint :
	ruff check --output-format=github .

.PHONY : format
format :
	black --check .

.PHONY : docformat
docformat :
	docformatter --config ./pyproject.toml --in-place $(SOURCE)

.PHONY : doctest-src
doctest-src :
	python -m pytest --xdoctest $(SOURCE)
	find . -type f -name "*.md" | xargs python -m doctest -o NORMALIZE_WHITESPACE -o ELLIPSIS -o REPORT_NDIFF

.PHONY : test
test :
	python -m pytest --xdoctest

.PHONY : unit-test
unit-test :
	python -m pytest --xdoctest --timeout 10 $(UNIT_TESTS)

.PHONY : unit-test-cov
unit-test-cov :
	python -m pytest --xdoctest --timeout 10 --cov-report html --cov-report xml --cov-report term --cov=$(NAME) $(UNIT_TESTS)

.PHONY : publish-pypi
publish-pypi :
	poetry config pypi-token.pypi ${PYPI_TOKEN}
	poetry publish --build

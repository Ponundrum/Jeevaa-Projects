.PHONY: install test lint selftest run figures clean

install:            ## install the package + dev tools
	pip install -e ".[dev]"

selftest:           ## trust the lab — analytic + Monte-Carlo checks 1-7 (~2s)
	python -c "from mmlab import self_test; self_test()"

test:               ## run the unit test suite (no network, ~4s)
	pytest -q

lint:               ## lint the package
	ruff check mmlab tests

run:                ## execute both notebooks end-to-end
	jupyter nbconvert --to notebook --execute --inplace notebooks/01_model_and_simulator.ipynb
	jupyter nbconvert --to notebook --execute --inplace notebooks/02_adverse_selection.ipynb

figures:            ## regenerate the committed docs/ figures
	python scripts/make_figures.py

clean:              ## remove caches and executed artifacts
	rm -rf **/__pycache__ .pytest_cache .ipynb_checkpoints

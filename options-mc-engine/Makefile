.PHONY: install test lint selftest run figures clean

install:            ## install the package + dev tools
	pip install -e ".[dev]"

selftest:           ## trust the engine — MC vs closed form (~1.3s)
	python -c "from qmc import self_test; self_test()"

test:               ## run the unit test suite (no network, ~3s)
	pytest -q

lint:               ## lint the package
	ruff check qmc tests

run:                ## execute both notebooks end-to-end
	jupyter nbconvert --to notebook --execute --inplace notebooks/01_mc_engine_and_validation.ipynb
	jupyter nbconvert --to notebook --execute --inplace notebooks/02_vol_surface_and_rough_vol.ipynb

figures:            ## regenerate the committed docs/ figures
	python scripts/make_figures.py

clean:              ## remove caches and executed artifacts
	rm -rf **/__pycache__ .pytest_cache .ipynb_checkpoints

.PHONY: install data test lint run figures clean

install:            ## install the package + dev tools
	pip install -e ".[dev]"

data:               ## prime the data cache (first run downloads ~25MB from Binance's archive)
	python -c "from qsa import Dataset; Dataset.load()"

test:               ## run the unit test suite (no network, <2s)
	pytest -q

lint:               ## lint the package
	ruff check qsa tests

run:                ## execute all three notebooks end-to-end
	jupyter nbconvert --to notebook --execute --inplace notebooks/01_combined_book.ipynb
	jupyter nbconvert --to notebook --execute --inplace notebooks/02_signal_research.ipynb
	jupyter nbconvert --to notebook --execute --inplace notebooks/03_walkforward.ipynb

figures:            ## regenerate the committed docs/ figures
	python scripts/make_figures.py

clean:              ## remove caches and executed artifacts
	rm -rf **/__pycache__ .pytest_cache .ipynb_checkpoints

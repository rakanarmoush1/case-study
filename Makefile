.PHONY: setup run smoke eda check evaluate train predict interpret monitor clean

setup:            ## Create a virtual environment and install the package
	python3 -m venv .venv
	.venv/bin/pip install -r requirements.txt
	.venv/bin/pip install -e .

run:              ## Full experiment with tuning (about 12 minutes on a laptop) -> reports/, models/
	fraud-detection run --tune --trials 25

smoke:            ## End-to-end pipeline on the first 48 steps (about a minute)
	fraud-detection run --quick --max-steps 48 --reports reports/smoke

eda:
	python scripts/eda.py

evaluate:
	python scripts/evaluate.py

train:
	python scripts/train.py

predict:
	python scripts/predict.py --input data/fraud.csv --output scored.csv --min-step 144

interpret:
	python scripts/interpret.py

check:
	python scripts/check_features.py

monitor:
	python scripts/monitor.py --input data/fraud.csv --min-step 144

clean:
	rm -rf reports/smoke scored.csv

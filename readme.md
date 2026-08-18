# Freight Rate Prediction Challenge

See `Freight_Rate_ML_Assessment.pdf` for the assessment instructions.

## What to do

1. Train and validate your model using `data/train_test.csv`.
2. Predict every load in `data/validation.csv`. Each load has a unique `load_id`.
3. Fill the matching `predicted_rate` values in `data/validation_predictions_template.csv` and save it as `validation_predictions.csv`.
4. Predict every row in `data/december_chart_inputs.csv` by filling its `predicted_rate` column.
5. Install the scorer requirements and run:

```bash
python -m pip install -r requirements.txt
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
```

The scorer validates both files and creates `scorer_results/candidate_december.png`.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate       # macOS/Linux
python -m pip install -r requirements.txt
```

## Train (notebook)

The full workflow — EDA, time-based validation, model selection, error analysis, and export — lives in:

`notebooks/freight_rate_modeling.ipynb`

Run it top-to-bottom. The final cells save trained artifacts to `models/`:

- `final_model.joblib` — fitted preprocessing + regressor pipeline
- `feature_stats.joblib` — imputation and coordinate lookups fit on Jan–Oct train
- `model_config.joblib` — feature list, log-target flag, and model name

## Predict (script)

After the notebook export cell has run (or after copying committed artifacts into `models/`):

```bash
python predict.py
```

This regenerates:

- `validation_predictions.csv` (12,000 holdout loads)
- `data/december_chart_inputs.csv` (31 December lane rows)

Optional paths:

```bash
python predict.py --model-dir models --validation-output validation_predictions.csv --december-output data/december_chart_inputs.csv
```

## Score

From the repo root:

```bash
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
```

Output: `scorer_results/candidate_december.png`

## Submit

- GitHub repository containing your code, dependencies, and run instructions
- `validation_predictions.csv`
- PDF or DOCX report containing your validation, data split approach and `candidate_december.png`
- 2–3 minute Loom link

## Project layout

| Path | Purpose |
|------|---------|
| `notebooks/freight_rate_modeling.ipynb` | End-to-end analysis and training |
| `freight_model.py` | Shared feature engineering and artifact I/O |
| `predict.py` | Load saved model and write submission CSVs |
| `models/` | Saved pipeline and preprocessing stats |
| `score.py` | Validates submissions and plots December curve |

#!/usr/bin/env python3
"""Regenerate submission CSVs from saved model artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from freight_model import DATA_DIR, engineer_features, load_artifacts, predict_rate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "models",
        help="Directory containing final_model.joblib and related files",
    )
    parser.add_argument(
        "--validation-output",
        type=Path,
        default=Path(__file__).resolve().parent / "validation_predictions.csv",
        help="Output path for holdout predictions",
    )
    parser.add_argument(
        "--december-output",
        type=Path,
        default=DATA_DIR / "december_chart_inputs.csv",
        help="Output path for December lane predictions",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model, feature_stats, config = load_artifacts(args.model_dir)
    log_target = config["log_target"]

    val_df = pd.read_csv(DATA_DIR / "validation.csv")
    december_df = pd.read_csv(DATA_DIR / "december_chart_inputs.csv")

    holdout_feat = engineer_features(val_df, feature_stats)
    holdout_preds = predict_rate(model, holdout_feat, log_target=log_target)

    validation_predictions = pd.DataFrame(
        {"load_id": val_df["load_id"], "predicted_rate": holdout_preds}
    )
    assert len(validation_predictions) == 12_000
    assert validation_predictions["load_id"].is_unique
    assert (validation_predictions["predicted_rate"] > 0).all()
    validation_predictions.to_csv(args.validation_output, index=False)

    december_submission = december_df.copy()
    december_feat = engineer_features(december_submission, feature_stats)
    december_submission["predicted_rate"] = predict_rate(
        model, december_feat, log_target=log_target
    )
    december_submission.to_csv(args.december_output, index=False)

    print(f"Wrote {args.validation_output} ({len(validation_predictions):,} rows)")
    print(f"Wrote {args.december_output} ({len(december_submission):,} rows)")
    print(
        "Run scorer:\n"
        f"python score.py --predictions {args.validation_output} "
        f"--december-predictions {args.december_output}"
    )


if __name__ == "__main__":
    main()

"""
run_prediction.py
==================
Reproduces the paper's Table 2 (accuracy by train/test split), Table 3
(mean/std/variance at each model's best split, from repeated random splits)
and Table 4 (top-10 predicted stocks per model), using the real dataset
supplied by the user and the models/hyperparameters defined in models.py.

Run: python3 run_prediction.py
Outputs (outputs/):
  table2_prediction_by_split.csv
  table3_prediction_summary.csv
  table4_top10_stocks.csv
  predictions_latest.csv   (full ranked predictions for every company, all models)
"""
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

from data import build_panel, latest_feature_snapshot
from models import MODEL_FACTORY, MODEL_FEATURE_ORDER

OUT = "outputs"
SPLITS = [0.1, 0.2, 0.3, 0.4]     # test-set fractions, matching the paper's Table 1/2
N_REPEATS = 5                      # repeated random splits at the best ratio, for Table 3
SEED = 42


def metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    return mae, mse, rmse


def run_split_table(panel):
    X = panel[MODEL_FEATURE_ORDER].values
    y = panel["target_return"].values

    rows = []
    for model_name, factory in MODEL_FACTORY.items():
        for frac in SPLITS:
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=frac, random_state=SEED)
            model = factory()
            model.fit(Xtr, ytr)
            pred = model.predict(Xte)
            mae, mse, rmse = metrics(yte, pred)
            rows.append({"Model": model_name, "Test set fraction": frac,
                         "MAE": mae, "MSE": mse, "RMSE": rmse})
            print(f"[split-table] {model_name:9s} frac={frac:.1f}  "
                  f"MAE={mae:.6f} MSE={mse:.6f} RMSE={rmse:.6f}")
    return pd.DataFrame(rows)


def best_split_per_model(table2):
    """Best split = lowest test RMSE, chosen empirically from our results
    (the paper's own best-split choice cannot be assumed to transfer to a
    different feature set built on different underlying data)."""
    idx = table2.groupby("Model")["RMSE"].idxmin()
    best = table2.loc[idx, ["Model", "Test set fraction"]].reset_index(drop=True)
    return dict(zip(best["Model"], best["Test set fraction"]))


def run_summary_table(panel, best_split):
    X = panel[MODEL_FEATURE_ORDER].values
    y = panel["target_return"].values

    rows = []
    for model_name, factory in MODEL_FACTORY.items():
        frac = best_split[model_name]
        maes, mses, rmses = [], [], []
        for rep in range(N_REPEATS):
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=frac, random_state=SEED + rep)
            model = factory()
            model.fit(Xtr, ytr)
            pred = model.predict(Xte)
            mae, mse, rmse = metrics(yte, pred)
            maes.append(mae); mses.append(mse); rmses.append(rmse)
        for stat_name, arr in [("mean", np.mean), ("sigma", np.std), ("sigma2", np.var)]:
            rows.append({
                "Model": model_name, "best_split_test_fraction": frac, "stat": stat_name,
                "MAE": arr(maes), "MSE": arr(mses), "RMSE": arr(rmses),
            })
        print(f"[summary] {model_name:9s} best_split={frac:.1f}  "
              f"mean RMSE={np.mean(rmses):.6f}  sigma RMSE={np.std(rmses):.6f}")
    return pd.DataFrame(rows)


def run_top10(panel, best_split, top_n=10):
    X = panel[MODEL_FEATURE_ORDER].values
    y = panel["target_return"].values
    latest = latest_feature_snapshot(panel)
    X_latest = latest[MODEL_FEATURE_ORDER].values

    all_preds = {"Company": latest["Company"].values}
    top_tables = {}
    for model_name, factory in MODEL_FACTORY.items():
        frac = best_split[model_name]
        # Train on the (1-frac) training portion of the *same* split used for
        # the accuracy tables, then predict forward for every company's most
        # recent feature snapshot -- this is the model's forward-looking
        # "candidate pool" prediction, analogous to the paper's Table 4.
        Xtr, _, ytr, _ = train_test_split(X, y, test_size=frac, random_state=SEED)
        model = factory()
        model.fit(Xtr, ytr)
        preds = model.predict(X_latest)
        all_preds[model_name] = preds

        ranked = latest.assign(pred_return=preds).sort_values("pred_return", ascending=False)
        top_tables[model_name] = ranked[["Company", "pred_return"]].head(top_n).reset_index(drop=True)
        print(f"[top10] {model_name}: {list(top_tables[model_name]['Company'])}")

    preds_df = pd.DataFrame(all_preds)

    top10_wide = pd.DataFrame({
        name: tbl["Company"].tolist() for name, tbl in top_tables.items()
    })
    return top10_wide, preds_df, top_tables


def main():
    import os
    os.makedirs(OUT, exist_ok=True)

    panel = build_panel()
    print(f"panel: {panel.shape[0]} samples, {panel['Company'].nunique()} companies")

    table2 = run_split_table(panel)
    table2.to_csv(f"{OUT}/table2_prediction_by_split.csv", index=False)

    best_split = best_split_per_model(table2)
    print("Best split per model (empirical, lowest test RMSE):", best_split)
    with open(f"{OUT}/best_split_per_model.json", "w") as f:
        json.dump(best_split, f, indent=2)

    table3 = run_summary_table(panel, best_split)
    table3.to_csv(f"{OUT}/table3_prediction_summary.csv", index=False)

    top10_wide, preds_df, top_tables = run_top10(panel, best_split)
    top10_wide.to_csv(f"{OUT}/table4_top10_stocks.csv", index=False)
    preds_df.to_csv(f"{OUT}/predictions_latest.csv", index=False)

    for name, tbl in top_tables.items():
        tbl.to_csv(f"{OUT}/top10_detail_{name}.csv", index=False)

    print("\nDone. Outputs written to", OUT)


if __name__ == "__main__":
    main()

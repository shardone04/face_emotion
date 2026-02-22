import argparse
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib


def aggregate_by_segment(df: pd.DataFrame, feature_cols):
    """
    Convert frame-level features into segment-level vectors via summary stats.
    Output: one row per segment_id.
    """
    grouped = df.groupby("segment_id")

    agg_list = []
    for seg_id, g in grouped:
        # Label = mode (most frequent label in the segment)
        label_id = int(g["label_id"].mode().iloc[0])
        label_name = g["label_name"].mode().iloc[0]

        feats = g[feature_cols]
        row = {
            "segment_id": seg_id,
            "label_id": label_id,
            "label_name": label_name,
        }

        # Stats: mean, std, min, max
        row.update({f"{c}_mean": feats[c].mean() for c in feature_cols})
        row.update({f"{c}_std": feats[c].std(ddof=0) for c in feature_cols})
        row.update({f"{c}_min": feats[c].min() for c in feature_cols})
        row.update({f"{c}_max": feats[c].max() for c in feature_cols})

        agg_list.append(row)

    return pd.DataFrame(agg_list)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", default="data/raw_features.csv")
    ap.add_argument("--model_out", default="models/rf_fer.joblib")
    ap.add_argument("--n_estimators", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)

    df = pd.read_csv(args.in_csv)
    feature_cols = [c for c in df.columns if any(x in c for x in ["_AoT", "_ICC", "_ICAT"])]

    seg_df = aggregate_by_segment(df, feature_cols)

    X_cols = [c for c in seg_df.columns if c.endswith(("_mean", "_std", "_min", "_max"))]
    X = seg_df[X_cols].values
    y = seg_df["label_id"].values
    groups = seg_df["segment_id"].values

    # GroupKFold: segment 단위 누수 방지
    gkf = GroupKFold(n_splits=min(5, len(seg_df)))
    preds = np.zeros_like(y)

    for fold, (tr, te) in enumerate(gkf.split(X, y, groups=groups), start=1):
        clf = RandomForestClassifier(
            n_estimators=args.n_estimators,
            random_state=args.seed,
            n_jobs=-1,
            class_weight="balanced_subsample",
        )
        clf.fit(X[tr], y[tr])
        preds[te] = clf.predict(X[te])
        print(f"Fold {fold} done.")

    print("\n=== CV Classification Report ===")
    print(classification_report(y, preds))
    print("Confusion Matrix:\n", confusion_matrix(y, preds))

    # Train final model on all data
    final_clf = RandomForestClassifier(
        n_estimators=args.n_estimators,
        random_state=args.seed,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    final_clf.fit(X, y)

    payload = {
        "model": final_clf,
        "feature_columns": X_cols,
    }
    joblib.dump(payload, args.model_out)
    print(f"\nSaved model to: {args.model_out}")


if __name__ == "__main__":
    main()
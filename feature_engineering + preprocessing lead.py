from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DEFAULT_TARGET = "ARR_DEL15"
LEAKAGE_COLUMNS = {
    "ARR_DELAY",
    "FLIGHT_STATUS",
    "IS_EXTREME_DELAY",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Feature engineering and preprocessing pipeline for the flight delay project."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to the cleaned CSV dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("person3_outputs"),
        help="Directory where all outputs will be written.",
    )
    parser.add_argument(
        "--target",
        type=str,
        default=DEFAULT_TARGET,
        help="Target column for the downstream supervised model.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test split ratio.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--prediction-stage",
        type=str,
        choices=["pre_departure", "post_departure"],
        default="post_departure",
        help="Whether the model is intended to predict before or after the flight departs.",
    )
    return parser.parse_args()


def hhmm_to_minutes(series: pd.Series) -> pd.Series:
    values = series.fillna(0).astype(int)
    hours = values // 100
    minutes = values % 100
    return hours * 60 + minutes


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.copy()

    enriched["ROUTE"] = (
        enriched["ORIGIN_AIRPORT_ID"].astype(str)
        + "_"
        + enriched["DEST_AIRPORT_ID"].astype(str)
    )

    flight_dates = pd.to_datetime(
        {
            "year": 2025,
            "month": enriched["MONTH"],
            "day": enriched["DAY_OF_MONTH"],
        }
    )
    enriched["DAY_OF_WEEK"] = flight_dates.dt.dayofweek
    enriched["IS_WEEKEND"] = enriched["DAY_OF_WEEK"].isin([5, 6]).astype(int)

    dep_minutes = hhmm_to_minutes(enriched["CRS_DEP_TIME"])
    arr_minutes = hhmm_to_minutes(enriched["CRS_ARR_TIME"])
    minutes_per_day = 24 * 60

    enriched["CRS_DEP_SIN"] = np.sin(2 * np.pi * dep_minutes / minutes_per_day)
    enriched["CRS_DEP_COS"] = np.cos(2 * np.pi * dep_minutes / minutes_per_day)
    enriched["CRS_ARR_SIN"] = np.sin(2 * np.pi * arr_minutes / minutes_per_day)
    enriched["CRS_ARR_COS"] = np.cos(2 * np.pi * arr_minutes / minutes_per_day)

    enriched["DISTANCE_BUCKET"] = pd.cut(
        enriched["DISTANCE"],
        bins=[-np.inf, 500, 1000, 2000, np.inf],
        labels=["short", "medium", "long", "very_long"],
    ).astype(str)

    return enriched


def build_feature_frame(
    df: pd.DataFrame, target: str, prediction_stage: str
) -> tuple[pd.DataFrame, pd.Series]:
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' was not found in the dataset.")

    working = df.copy()
    working = working[working[target].notna()].copy()
    working[target] = working[target].astype(int)

    drop_columns = {target, *LEAKAGE_COLUMNS}
    if prediction_stage == "pre_departure":
        drop_columns.add("DEP_DELAY")
    feature_columns = [col for col in working.columns if col not in drop_columns]
    X = working[feature_columns]
    y = working[target]
    return X, y


def correlation_report(df: pd.DataFrame, threshold: float = 0.9) -> dict[str, object]:
    numeric_df = df.select_dtypes(include=["number"]).copy()
    if numeric_df.empty:
        return {"threshold": threshold, "high_correlation_pairs": []}

    corr = numeric_df.corr().abs()
    pairs: list[dict[str, object]] = []
    cols = corr.columns.tolist()
    for i, left in enumerate(cols):
        for right in cols[i + 1 :]:
            value = corr.loc[left, right]
            if pd.notna(value) and value >= threshold:
                pairs.append(
                    {
                        "feature_1": left,
                        "feature_2": right,
                        "correlation": round(float(value), 4),
                    }
                )
    return {"threshold": threshold, "high_correlation_pairs": pairs}


def correlated_drop_list(df: pd.DataFrame, threshold: float = 0.9) -> tuple[list[str], list[dict[str, object]]]:
    report = correlation_report(df, threshold=threshold)
    to_drop: list[str] = []
    seen: set[str] = set()
    for pair in report["high_correlation_pairs"]:
        candidate = pair["feature_2"]
        if candidate not in seen:
            to_drop.append(candidate)
            seen.add(candidate)
    return to_drop, report["high_correlation_pairs"]


def make_preprocessor(X: pd.DataFrame) -> tuple[ColumnTransformer, list[str], list[str]]:
    categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_features = [
        col for col in X.columns if col not in categorical_features
    ]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=True),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        sparse_threshold=0.3,
    )

    return preprocessor, numeric_features, categorical_features


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    try:
        return preprocessor.get_feature_names_out().tolist()
    except AttributeError:
        names: list[str] = []
        for name, transformer, columns in preprocessor.transformers_:
            if name == "remainder":
                continue
            if hasattr(transformer, "get_feature_names_out"):
                names.extend(transformer.get_feature_names_out(columns).tolist())
            elif hasattr(transformer, "named_steps") and "encoder" in transformer.named_steps:
                encoder = transformer.named_steps["encoder"]
                names.extend(encoder.get_feature_names_out(columns).tolist())
            else:
                names.extend(columns)
        return names


def ensure_sparse(matrix) -> sparse.csr_matrix:
    if sparse.issparse(matrix):
        return matrix.tocsr()
    return sparse.csr_matrix(matrix)


def save_series(series: pd.Series, path: Path) -> None:
    series.to_frame(name=series.name or "target").to_csv(path, index=False)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)
    engineered = add_engineered_features(df)
    X, y = build_feature_frame(engineered, args.target, args.prediction_stage)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    correlated_drops, high_corr_pairs = correlated_drop_list(X_train, threshold=0.9)
    if correlated_drops:
        X_train = X_train.drop(columns=correlated_drops, errors="ignore")
        X_test = X_test.drop(columns=correlated_drops, errors="ignore")

    preprocessor, numeric_features, categorical_features = make_preprocessor(X_train)
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    selector = VarianceThreshold(threshold=0.0)
    X_train_selected = selector.fit_transform(X_train_processed)
    X_test_selected = selector.transform(X_test_processed)

    feature_names = get_feature_names(preprocessor)
    selected_feature_names = [
        name for name, keep in zip(feature_names, selector.get_support()) if keep
    ]

    sparse.save_npz(
        args.output_dir / "X_train_processed.npz", ensure_sparse(X_train_selected)
    )
    sparse.save_npz(
        args.output_dir / "X_test_processed.npz", ensure_sparse(X_test_selected)
    )

    X_train.to_csv(args.output_dir / "train_raw.csv", index=False)
    X_test.to_csv(args.output_dir / "test_raw.csv", index=False)
    save_series(y_train.rename(args.target), args.output_dir / "y_train.csv")
    save_series(y_test.rename(args.target), args.output_dir / "y_test.csv")

    pipeline_bundle = {
        "preprocessor": preprocessor,
        "variance_selector": selector,
        "selected_feature_names": selected_feature_names,
        "target_column": args.target,
    }
    joblib.dump(pipeline_bundle, args.output_dir / "preprocessing_pipeline.joblib")

    summary = {
        "input_file": str(args.input),
        "target_column": args.target,
        "prediction_stage": args.prediction_stage,
        "row_count_after_target_filter": int(len(X)),
        "train_shape_raw": [int(X_train.shape[0]), int(X_train.shape[1])],
        "test_shape_raw": [int(X_test.shape[0]), int(X_test.shape[1])],
        "train_shape_processed": [
            int(X_train_selected.shape[0]),
            int(X_train_selected.shape[1]),
        ],
        "test_shape_processed": [
            int(X_test_selected.shape[0]),
            int(X_test_selected.shape[1]),
        ],
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "engineered_features": [
            "ROUTE",
            "DAY_OF_WEEK",
            "IS_WEEKEND",
            "CRS_DEP_SIN",
            "CRS_DEP_COS",
            "CRS_ARR_SIN",
            "CRS_ARR_COS",
            "DISTANCE_BUCKET",
        ],
        "dropped_leakage_columns": sorted(
            col
            for col in (LEAKAGE_COLUMNS | ({ "DEP_DELAY" } if args.prediction_stage == "pre_departure" else set()))
            if col in engineered.columns
        ),
        "dropped_for_multicollinearity": correlated_drops,
        "high_correlation_pairs": high_corr_pairs,
    }
    with (args.output_dir / "feature_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Feature engineering and preprocessing complete.")
    print(f"Outputs written to: {args.output_dir.resolve()}")
    print(
        "Processed feature shapes:",
        X_train_selected.shape,
        X_test_selected.shape,
    )


if __name__ == "__main__":
    main()

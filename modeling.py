"""Notebook-faithful data preparation and model training for the hotel lab."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42
TARGET = "is_canceled"
DATA_PATH = Path(__file__).parent / "data" / "hotel_bookings.zip"

ENGINEERED_NUMERIC = ("total_nights", "total_guests", "arrival_month_num")
ENGINEERED_CATEGORICAL = ("arrival_season", "lead_time_category")
ENGINEERED_FEATURES = set(ENGINEERED_NUMERIC + ENGINEERED_CATEGORICAL)
EXCLUDED_COLUMNS = {
    TARGET,
    "reservation_status",
    "reservation_status_date",
    "assigned_room_type",
    "agent",
    "company",
    "arrival_date_month",
}

MONTHS = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)


def _season(month: int | None) -> str:
    if month is None:
        return "Unknown"
    if month in (12, 1, 2):
        return "Winter"
    if month in (3, 4, 5):
        return "Spring"
    if month in (6, 7, 8):
        return "Summer"
    return "Autumn"


def engineer_booking_features(raw_booking: dict[str, Any]) -> dict[str, Any]:
    """Apply the notebook's booking-derived features to one user-entered booking."""
    booking = dict(raw_booking)
    month_value = booking.pop("arrival_date_month", "January")
    month_number = {month: i for i, month in enumerate(MONTHS, start=1)}.get(
        str(month_value)
    )
    lead_time = booking.get("lead_time", 0)
    booking["total_nights"] = (
        booking.get("stays_in_weekend_nights", 0)
        + booking.get("stays_in_week_nights", 0)
    )
    booking["total_guests"] = (
        booking.get("adults", 0)
        + booking.get("children", 0)
        + booking.get("babies", 0)
    )
    booking["arrival_month_num"] = month_number
    booking["arrival_season"] = _season(month_number)
    booking["lead_time_category"] = pd.cut(
        [lead_time],
        bins=[-1, 7, 30, 90, 180, np.inf],
        labels=["0-7 days", "8-30 days", "31-90 days", "91-180 days", "181+ days"],
    ).astype(object)[0]
    return booking


def load_and_prepare_data(path: Path = DATA_PATH) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Load the original CSV-in-ZIP and reproduce notebook cleaning and features."""
    df = pd.read_csv(path, compression="zip")
    df = df.drop_duplicates().copy()
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df[df[TARGET].isin([0, 1])].copy()
    df["children"] = pd.to_numeric(df["children"], errors="coerce")
    df["reservation_status_date"] = pd.to_datetime(
        df["reservation_status_date"], errors="coerce"
    )
    df = df[df["lead_time"] >= 0].copy()
    df = df[df["adr"] >= 0].copy()
    df = df[
        (df["stays_in_weekend_nights"] >= 0)
        & (df["stays_in_week_nights"] >= 0)
    ].copy()

    month_map = {month: i for i, month in enumerate(MONTHS, start=1)}
    df["arrival_month_num"] = df["arrival_date_month"].map(month_map)
    df["arrival_season"] = df["arrival_month_num"].apply(
        lambda value: _season(None if pd.isna(value) else int(value))
    )
    df["lead_time_category"] = pd.cut(
        df["lead_time"],
        bins=[-1, 7, 30, 90, 180, np.inf],
        labels=["0-7 days", "8-30 days", "31-90 days", "91-180 days", "181+ days"],
    ).astype(object)
    df["total_nights"] = (
        df["stays_in_weekend_nights"] + df["stays_in_week_nights"]
    )
    df["total_guests"] = (
        df["adults"].fillna(0) + df["children"].fillna(0) + df["babies"].fillna(0)
    )

    model_df = df.drop(columns=list(EXCLUDED_COLUMNS - {TARGET}), errors="ignore")
    X = model_df.drop(columns=[TARGET])
    y = model_df[TARGET].astype(int)
    return df, X, y


def _make_preprocessor(
    X_train: pd.DataFrame,
) -> tuple[ColumnTransformer, list[str], list[str], list[str], list[str]]:
    numeric_features = X_train.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X_train.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()
    mean_features: list[str] = []
    median_features: list[str] = []
    for feature in numeric_features:
        series = X_train[feature].dropna()
        skew_value = series.skew() if len(series) > 1 else 0
        (median_features if abs(skew_value) > 1 else mean_features).append(feature)

    transformers: list[tuple[str, Any, list[str]]] = []
    if mean_features:
        transformers.append(
            (
                "mean_numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="mean")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                mean_features,
            )
        )
    if median_features:
        transformers.append(
            (
                "median_numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                median_features,
            )
        )
    if categorical_features:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            )
        )
    return (
        ColumnTransformer(transformers=transformers),
        numeric_features,
        categorical_features,
        mean_features,
        median_features,
    )


def fit_models(path: Path = DATA_PATH) -> dict[str, Any]:
    """Train the three exact notebook pipelines and calculate held-out outputs."""
    df, X, y = load_and_prepare_data(path)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    preprocessor, numeric, categorical, mean_features, median_features = (
        _make_preprocessor(X_train)
    )
    models: dict[str, Pipeline] = {
        "Logistic Regression": Pipeline(
            [
                ("preprocessor", preprocessor),
                (
                    "model",
                    LogisticRegression(
                        max_iter=2000, class_weight=None, random_state=RANDOM_STATE
                    ),
                ),
            ]
        ),
        "Decision Tree": Pipeline(
            [
                ("preprocessor", preprocessor),
                (
                    "model",
                    DecisionTreeClassifier(
                        max_depth=12, min_samples_leaf=20, random_state=RANDOM_STATE
                    ),
                ),
            ]
        ),
        "Random Forest": Pipeline(
            [
                ("preprocessor", preprocessor),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=150,
                        max_depth=18,
                        min_samples_leaf=5,
                        n_jobs=-1,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }
    for pipeline in models.values():
        pipeline.fit(X_train, y_train)

    probabilities = {
        name: pipeline.predict_proba(X_test)[:, 1]
        for name, pipeline in models.items()
    }
    bundle: dict[str, Any] = {
        "df": df,
        "X": X,
        "y": y,
        "X_train": X_train,
        "X_test": X_test,
        "y_test": y_test,
        "models": models,
        "probabilities": probabilities,
        "feature_names": X.columns.tolist(),
        "numeric_features": numeric,
        "categorical_features": categorical,
        "mean_features": mean_features,
        "median_features": median_features,
        "roc": {
            name: roc_curve(y_test, scores)
            for name, scores in probabilities.items()
        },
        "train_rows": len(X_train),
        "test_rows": len(X_test),
    }
    bundle["base_metrics"] = metrics_at_threshold(bundle, 0.5)
    return bundle


def metrics_at_threshold(bundle: dict[str, Any], threshold: float) -> dict[str, dict[str, Any]]:
    """Recalculate all threshold-dependent test metrics from actual probabilities."""
    metrics: dict[str, dict[str, Any]] = {}
    for name, scores in bundle["probabilities"].items():
        predicted = (scores >= threshold).astype(int)
        truth = bundle["y_test"]
        metrics[name] = {
            "Accuracy": accuracy_score(truth, predicted),
            "Precision": precision_score(truth, predicted, zero_division=0),
            "Recall": recall_score(truth, predicted, zero_division=0),
            "F1": f1_score(truth, predicted, zero_division=0),
            "ROC-AUC": roc_auc_score(truth, scores),
            "Confusion": confusion_matrix(truth, predicted, labels=[0, 1]),
        }
    return metrics


def transform_booking(pipeline: Pipeline, booking: dict[str, Any]) -> Any:
    """Transform a user booking through the fitted original preprocessor."""
    row = pd.DataFrame([engineer_booking_features(booking)])
    row = row.reindex(columns=pipeline.named_steps["preprocessor"].feature_names_in_)
    return pipeline.named_steps["preprocessor"].transform(row)


def booking_frame(
    booking: dict[str, Any], feature_names: list[str]
) -> pd.DataFrame:
    """Build a model-ready row in the training feature order."""
    engineered = engineer_booking_features(booking)
    return pd.DataFrame([engineered]).reindex(columns=feature_names)


def tree_feature_descriptions(pipeline: Pipeline) -> list[dict[str, Any]]:
    """Describe transformed tree features and invert scaled numeric thresholds."""
    preprocessor = pipeline.named_steps["preprocessor"]
    transformed_names = preprocessor.get_feature_names_out()
    descriptions: list[dict[str, Any]] = []
    for name in transformed_names:
        if name.startswith(("mean_numeric__", "median_numeric__")):
            group, raw_name = name.split("__", 1)
            transformer = next(item[1] for item in preprocessor.transformers_ if item[0] == group)
            columns = list(next(item[2] for item in preprocessor.transformers_ if item[0] == group))
            position = columns.index(raw_name)
            scaler = transformer.named_steps["scaler"]
            descriptions.append(
                {
                    "kind": "numeric",
                    "raw_name": raw_name,
                    "label": raw_name.replace("_", " "),
                    "scale": float(scaler.scale_[position]),
                    "mean": float(scaler.mean_[position]),
                }
            )
        elif name.startswith("cat__"):
            encoded = name.split("__", 1)[1]
            cat_columns = list(
                next(item[2] for item in preprocessor.transformers_ if item[0] == "cat")
            )
            matching = [column for column in cat_columns if encoded.startswith(f"{column}_")]
            raw_name = max(matching, key=len) if matching else encoded
            category = encoded[len(raw_name) + 1 :] if raw_name != encoded else ""
            descriptions.append(
                {
                    "kind": "categorical",
                    "raw_name": raw_name,
                    "category": category,
                    "label": raw_name.replace("_", " "),
                }
            )
        else:
            descriptions.append(
                {"kind": "other", "raw_name": str(name), "label": str(name)}
            )
    return descriptions


def serialize_tree(
    pipeline: Pipeline,
    estimator: DecisionTreeClassifier,
    max_depth: int = 5,
    threshold: float = 0.5,
) -> list[dict[str, Any]]:
    """Export actual nodes from a fitted sklearn tree, up to a visual depth."""
    structure = estimator.tree_
    descriptions = tree_feature_descriptions(pipeline)
    classes = list(estimator.classes_)
    nodes: list[dict[str, Any]] = []
    stack = [(0, 0)]
    while stack:
        node_id, depth = stack.pop()
        if depth > max_depth:
            continue
        left = int(structure.children_left[node_id])
        right = int(structure.children_right[node_id])
        leaf = left == right
        samples = int(structure.n_node_samples[node_id])
        values = np.asarray(structure.value[node_id][0], dtype=float)
        if values.sum() <= 1.000001:
            counts = values * samples
        else:
            counts = values
        proportions = counts / counts.sum() if counts.sum() else np.zeros_like(counts)
        cancellation_probability = (
            float(proportions[classes.index(1)]) if 1 in classes else 0.0
        )
        prediction = int(cancellation_probability >= threshold)
        feature_description: dict[str, Any] | None = None
        split_threshold = None
        condition = "Leaf decision"
        feature_label = "Outcome"
        if not leaf:
            feature_description = descriptions[int(structure.feature[node_id])]
            split_threshold = float(structure.threshold[node_id])
            if feature_description["kind"] == "numeric":
                raw_threshold = (
                    split_threshold * feature_description["scale"]
                    + feature_description["mean"]
                )
                condition = (
                    f"{feature_description['label']} ≤ {raw_threshold:.2f}"
                )
                feature_label = feature_description["label"]
                split_threshold = raw_threshold
            else:
                category = feature_description.get("category", "")
                feature_label = feature_description["label"]
                condition = f"{feature_label} is {category}"
        nodes.append(
            {
                "id": node_id,
                "depth": depth,
                "left": left if not leaf else None,
                "right": right if not leaf else None,
                "leaf": leaf,
                "samples": samples,
                "gini": float(structure.impurity[node_id]),
                "counts": [int(round(value)) for value in counts],
                "class_labels": [str(value) for value in classes],
                "cancel_probability": cancellation_probability,
                "prediction": prediction,
                "threshold": threshold,
                "feature": feature_label,
                "condition": condition,
                "feature_kind": (
                    feature_description["kind"] if feature_description else "leaf"
                ),
                "category": (
                    feature_description.get("category")
                    if feature_description
                    else None
                ),
                "threshold": split_threshold,
                "classification_threshold": threshold,
            }
        )
        if not leaf and depth < max_depth:
            stack.append((right, depth + 1))
            stack.append((left, depth + 1))
    return nodes


def trace_tree(
    pipeline: Pipeline, estimator: DecisionTreeClassifier, row: pd.DataFrame
) -> tuple[list[int], list[dict[str, Any]]]:
    """Follow a row through its real, preprocessed decision tree."""
    transformed = pipeline.named_steps["preprocessor"].transform(row)
    values = (
        transformed.getrow(0).toarray()[0]
        if sparse.issparse(transformed)
        else np.asarray(transformed[0]).ravel()
    )
    structure = estimator.tree_
    descriptions = tree_feature_descriptions(pipeline)
    node_id = 0
    path = [node_id]
    steps: list[dict[str, Any]] = []
    while structure.children_left[node_id] != structure.children_right[node_id]:
        feature_index = int(structure.feature[node_id])
        threshold = float(structure.threshold[node_id])
        description = descriptions[feature_index]
        value = float(values[feature_index])
        goes_left = value <= threshold
        if description["kind"] == "numeric":
            raw_threshold = threshold * description["scale"] + description["mean"]
            condition = f"{description['label']} ≤ {raw_threshold:.2f}"
            branch = "LEFT: booking value is at or below the threshold" if goes_left else "RIGHT: booking value is above the threshold"
        else:
            category = description.get("category", "")
            condition = f"{description['label']} is {category}"
            branch = (
                f"LEFT: booking is not {category}"
                if goes_left
                else f"RIGHT: booking is {category}"
            )
        next_node = int(
            structure.children_left[node_id]
            if goes_left
            else structure.children_right[node_id]
        )
        steps.append(
            {
                "node": node_id,
                "condition": condition,
                "branch": branch,
                "next_node": next_node,
            }
        )
        node_id = next_node
        path.append(node_id)
    return path, steps


def cancellation_probability(pipeline: Pipeline, row: pd.DataFrame) -> float:
    """Return the fitted pipeline's cancellation probability for one booking."""
    return float(pipeline.predict_proba(row)[0, 1])

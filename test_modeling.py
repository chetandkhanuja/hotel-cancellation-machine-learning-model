import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from modeling import (
    engineer_booking_features,
    load_and_prepare_data,
    metrics_at_threshold,
    serialize_tree,
)


def test_booking_engineering_matches_notebook_bins_and_derived_fields():
    booking = engineer_booking_features(
        {
            "arrival_date_month": "December",
            "lead_time": 30,
            "stays_in_weekend_nights": 2,
            "stays_in_week_nights": 3,
            "adults": 2,
            "children": 1,
            "babies": 0,
        }
    )

    assert booking["total_nights"] == 5
    assert booking["total_guests"] == 3
    assert booking["arrival_month_num"] == 12
    assert booking["arrival_season"] == "Winter"
    assert booking["lead_time_category"] == "8-30 days"
    assert "arrival_date_month" not in booking


def test_shared_threshold_changes_labels_but_not_roc_auc():
    bundle = {
        "y_test": np.array([0, 0, 1, 1]),
        "probabilities": {
            "Example": np.array([0.1, 0.4, 0.45, 0.9]),
        },
    }

    low = metrics_at_threshold(bundle, 0.4)["Example"]
    high = metrics_at_threshold(bundle, 0.5)["Example"]

    assert low["Recall"] == 1.0
    assert high["Recall"] == 0.5
    assert low["ROC-AUC"] == high["ROC-AUC"]
    assert low["Confusion"].tolist() == [[1, 1], [0, 2]]
    assert high["Confusion"].tolist() == [[2, 0], [1, 1]]


def test_original_dataset_cleaning_matches_notebook_row_count():
    df, _, _ = load_and_prepare_data()

    assert len(df) == 87_395
    assert not df.duplicated().any()
    assert df["adr"].ge(0).all()
    assert df["lead_time"].ge(0).all()
    assert df["stays_in_weekend_nights"].ge(0).all()
    assert df["stays_in_week_nights"].ge(0).all()


def test_serialized_leaf_decision_uses_shared_threshold():
    X = pd.DataFrame({"lead_time": [1, 1, 1]})
    y = np.array([0, 1, 1])
    preprocessor = ColumnTransformer(
        [
            (
                "mean_numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="mean")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                ["lead_time"],
            )
        ]
    )
    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", DecisionTreeClassifier(random_state=42)),
        ]
    ).fit(X, y)
    estimator = pipeline.named_steps["model"]

    standard_cutoff = serialize_tree(pipeline, estimator, threshold=0.5)[0]
    strict_cutoff = serialize_tree(pipeline, estimator, threshold=0.7)[0]

    assert standard_cutoff["leaf"]
    assert standard_cutoff["cancel_probability"] == 2 / 3
    assert standard_cutoff["prediction"] == 1
    assert strict_cutoff["prediction"] == 0

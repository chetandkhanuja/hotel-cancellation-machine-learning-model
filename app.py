from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from components import render_forest, render_tree
from modeling import (
    DATA_PATH,
    ENGINEERED_FEATURES,
    MONTHS,
    booking_frame,
    fit_models,
    metrics_at_threshold,
    serialize_tree,
    trace_tree,
)

st.set_page_config(
    page_title="AURELIA · Hotel Intelligence",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
:root{--ink:#071821;--panel:#0d222b;--panel-2:#102a32;--line:#24414a;--muted:#91aaad;--text:#e9f2ef;--emerald:#35d49a;--teal:#0d7960}
.stApp{background:radial-gradient(ellipse at 68% -12%,#143c40 0%,#091c24 38%,#07151d 74%);color:var(--text);font-family:'DM Sans','Segoe UI',sans-serif}
[data-testid="stHeader"]{background:transparent}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#0b2028,#08161d);border-right:1px solid #1e3941}
[data-testid="stSidebar"] *{font-family:'DM Sans','Segoe UI',sans-serif}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p{color:#d5e5e1}
h1,h2,h3{font-family:'Manrope','DM Sans',sans-serif;letter-spacing:-.035em;color:#eff8f5}
h1{font-size:2.1rem!important;font-weight:800!important;margin-bottom:.2rem!important}
h2{font-size:1.35rem!important;font-weight:750!important}
h3{font-size:1.05rem!important;font-weight:700!important}
p,li{color:#b4c6c5}
.block-container{padding-top:2rem;padding-bottom:3rem;max-width:1450px}
div[data-testid="stMetric"]{background:linear-gradient(145deg,#102a32,#0b1c24);border:1px solid #24424a;border-radius:16px;padding:15px 17px;box-shadow:0 12px 30px #02090d33}
div[data-testid="stMetric"] label{color:#90aaad!important}
div[data-testid="stMetric"] [data-testid="stMetricValue"]{color:#eaf6f1!important;font-family:'Manrope',sans-serif}
div[data-testid="stMetric"] [data-testid="stMetricDelta"]{color:#56dba7!important}
.stButton>button,.stDownloadButton>button{border-radius:10px;border:1px solid #2d655a;background:#0d6f58;color:white;font-weight:700;transition:all .2s}
.stButton>button:hover,.stDownloadButton>button:hover{background:#118366;border-color:#42dba5;transform:translateY(-1px)}
.stSelectbox label,.stSlider label,.stNumberInput label{color:#b8cdca!important}
.stSlider [data-baseweb="slider"] div{background-color:#2bc68f}
.stRadio label{color:#d9e8e4!important}
div[data-testid="stExpander"]{border:1px solid #24414a;border-radius:12px;background:#0b1c24}
div[data-testid="stDataFrame"]{border:1px solid #24414a;border-radius:12px;overflow:hidden}
[data-testid="stPlotlyChart"]{border:1px solid #1d3941;border-radius:15px;background:#0a1c24;padding:6px}
.hero{padding:10px 0 17px;border-bottom:1px solid #203a42;margin-bottom:20px}
.eyebrow{font-size:11px;text-transform:uppercase;letter-spacing:.18em;color:#5edbaa;font-weight:800}
.subhead{color:#9db6b6;font-size:14px;margin-top:6px}
.booking-card{background:linear-gradient(125deg,#0c2b31,#102d32 60%,#123b35);border:1px solid #28554a;border-radius:17px;padding:15px 18px;margin:6px 0 16px}
.booking-card .small{color:#9ab6b1;font-size:11px;text-transform:uppercase;letter-spacing:.12em}
.booking-card .main{font-size:22px;font-weight:800;color:#eff8f2;margin:3px 0}
.booking-card .detail{font-size:12px;color:#a8c0bb}
.note-card{padding:13px 15px;border:1px solid #24414a;border-radius:13px;background:#0b1c24;color:#a9bfbd;line-height:1.5}
.insight{padding:14px 16px;background:linear-gradient(120deg,#0e2b30,#0c2028);border:1px solid #285148;border-radius:13px;color:#c5d9d4;line-height:1.55}
.insight b{color:#64e3af}
.stAlert{border-radius:12px}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def get_bundle() -> dict[str, Any]:
    return fit_models(DATA_PATH)


with st.spinner("Preparing the notebook's data and fitting its three models…"):
    bundle = get_bundle()

PAGES = (
    "Decision Tree Lab",
    "Logistic Regression Lab",
    "Random Forest Lab",
    "Model Comparison Lab",
)
CORE_FIELDS = (
    "hotel",
    "lead_time",
    "stays_in_weekend_nights",
    "stays_in_week_nights",
    "adults",
    "children",
    "meal",
    "market_segment",
    "distribution_channel",
    "deposit_type",
    "customer_type",
    "adr",
)
COUNT_FIELDS = {
    "lead_time",
    "arrival_date_year",
    "arrival_date_week_number",
    "arrival_date_day_of_month",
    "stays_in_weekend_nights",
    "stays_in_week_nights",
    "adults",
    "children",
    "babies",
    "is_repeated_guest",
    "previous_cancellations",
    "previous_bookings_not_canceled",
    "booking_changes",
    "days_in_waiting_list",
    "required_car_parking_spaces",
    "total_of_special_requests",
}


def _raw_fields() -> list[str]:
    return [
        name
        for name in bundle["feature_names"]
        if name not in ENGINEERED_FEATURES
    ] + ["arrival_date_month"]


def _booking_defaults() -> dict[str, Any]:
    training = bundle["X_train"]
    defaults: dict[str, Any] = {}
    for field in _raw_fields():
        if field == "arrival_date_month":
            defaults[field] = str(bundle["df"]["arrival_date_month"].mode().iloc[0])
        elif pd.api.types.is_numeric_dtype(training[field]):
            value = training[field].median()
            defaults[field] = int(round(value)) if field in COUNT_FIELDS else float(value)
        else:
            defaults[field] = training[field].mode(dropna=True).iloc[0]
    return defaults


if "booking_values" not in st.session_state:
    st.session_state.booking_values = _booking_defaults()
if "cancellation_threshold" not in st.session_state:
    st.session_state.cancellation_threshold = 0.50

st.sidebar.markdown(
    """
<div style="padding:9px 3px 14px">
 <div style="font-size:10px;letter-spacing:.22em;color:#55d9a8;font-weight:800">A U R E L I A</div>
 <div style="font-size:19px;font-weight:800;color:#eff8f5;margin-top:3px">Hotel Intelligence</div>
 <div style="font-size:11px;color:#87a4a7;margin-top:3px">CANCELLATION MODEL LAB</div>
</div>
""",
    unsafe_allow_html=True,
)
page = st.sidebar.radio("NAVIGATION", PAGES, label_visibility="collapsed")
st.sidebar.markdown("---")
st.sidebar.markdown("#### Shared decision cutoff")
threshold = st.sidebar.slider(
    "Cancellation probability threshold",
    min_value=0.0,
    max_value=1.0,
    step=0.01,
    key="cancellation_threshold",
    help="Changes the predicted class across all labs, not the underlying probability.",
)
st.sidebar.caption(f"Bookings at or above **{threshold:.0%}** are classified as likely cancellations.")
st.sidebar.markdown("---")
st.sidebar.markdown("#### Booking simulator")
st.sidebar.caption("Set one reservation; every model uses this same booking and fitted preprocessing.")

raw_fields = _raw_fields()
for field, value in st.session_state.booking_values.items():
    key = f"booking_{field}"
    if key not in st.session_state:
        st.session_state[key] = value

with st.sidebar.form("booking_simulator"):
    booking_inputs: dict[str, Any] = {}
    for field in CORE_FIELDS:
        if field not in raw_fields:
            continue
        key = f"booking_{field}"
        if field == "hotel":
            options = ["City Hotel", "Resort Hotel"]
            booking_inputs[field] = st.selectbox(
                "Hotel type", options, key=key
            )
        elif field in ("meal", "market_segment", "distribution_channel", "deposit_type", "customer_type"):
            options = sorted(bundle["X_train"][field].dropna().unique().tolist(), key=str)
            booking_inputs[field] = st.selectbox(
                field.replace("_", " ").title(), options, key=key
            )
        elif field in COUNT_FIELDS:
            values = bundle["X_train"][field].dropna()
            minimum, maximum = int(values.min()), int(values.max())
            booking_inputs[field] = st.number_input(
                field.replace("_", " ").title(),
                min_value=minimum,
                max_value=max(maximum, minimum + 1),
                step=1,
                key=key,
            )
        else:
            values = bundle["X_train"][field].dropna()
            minimum, maximum = float(values.min()), float(values.max())
            booking_inputs[field] = st.number_input(
                "Average daily rate (ADR)",
                min_value=minimum,
                max_value=max(maximum, minimum + 1.0),
                step=1.0,
                key=key,
                format="%.2f",
            )
    booking_inputs["arrival_date_month"] = st.selectbox(
        "Arrival month",
        list(MONTHS),
        key="booking_arrival_date_month",
    )
    advanced_fields = [
        field
        for field in raw_fields
        if field not in CORE_FIELDS and field != "arrival_date_month"
    ]
    with st.expander("More booking features", expanded=False):
        for field in advanced_fields:
            key = f"booking_{field}"
            if field in bundle["numeric_features"]:
                values = bundle["X_train"][field].dropna()
                minimum, maximum = float(values.min()), float(values.max())
                if field in COUNT_FIELDS:
                    booking_inputs[field] = st.number_input(
                        field.replace("_", " ").title(),
                        min_value=int(minimum),
                        max_value=max(int(maximum), int(minimum) + 1),
                        step=1,
                        key=key,
                    )
                else:
                    booking_inputs[field] = st.number_input(
                        field.replace("_", " ").title(),
                        min_value=minimum,
                        max_value=max(maximum, minimum + 1.0),
                        step=1.0,
                        key=key,
                        format="%.2f",
                    )
            else:
                options = sorted(
                    bundle["X_train"][field].dropna().unique().tolist(), key=str
                )
                booking_inputs[field] = st.selectbox(
                    field.replace("_", " ").title(), options, key=key
                )
    apply_booking = st.form_submit_button(
        "Apply booking to all labs", width="stretch"
    )

if apply_booking:
    st.session_state.booking_values = booking_inputs.copy()

booking = st.session_state.booking_values
row = booking_frame(booking, bundle["feature_names"])
metrics = metrics_at_threshold(bundle, threshold)


def hero(number: str, title: str, subtitle: str) -> None:
    st.markdown(
        f"""
<div class="hero"><div class="eyebrow">LIVE MODEL SIMULATION · {number}</div>
<h1>{title}</h1><div class="subhead">{subtitle}</div></div>
""",
        unsafe_allow_html=True,
    )


def booking_banner(probability: float, model_name: str) -> None:
    predicted = "Likely to cancel" if probability >= threshold else "Likely to stay"
    st.markdown(
        f"""
<div class="booking-card"><div class="small">Shared reservation · {model_name}</div>
<div class="main">{predicted} <span style="color:#63e1b0">· {probability:.1%}</span></div>
<div class="detail">{booking['hotel']} · {booking['lead_time']} days lead ·
{booking['stays_in_weekend_nights'] + booking['stays_in_week_nights']} nights ·
{booking['adults'] + booking['children'] + booking['babies']} guests ·
cutoff {threshold:.0%}</div></div>
""",
        unsafe_allow_html=True,
    )


def _tree_stats(estimator: Any) -> tuple[int, int, int]:
    tree = estimator.tree_
    depth = int(estimator.get_depth())
    return depth, int(tree.node_count), int(np.sum(tree.children_left == -1))


def _path_expander(
    name: str, estimator: Any, model_pipeline: Any, probability: float
) -> None:
    path, steps = trace_tree(model_pipeline, estimator, row)
    with st.expander(f"Full booking path · {len(steps)} split decisions · leaf {path[-1]}"):
        if not steps:
            st.write("The root is a leaf; no split was needed.")
        for index, step in enumerate(steps, start=1):
            st.markdown(
                f"**{index}. Node {step['node']}** — `{step['condition']}`  \n"
                f"→ {step['branch']} → node {step['next_node']}"
            )
        prediction = "cancel" if probability >= threshold else "not cancel"
        st.markdown(
            f"**Leaf {path[-1]}:** cancellation probability **{probability:.1%}**; "
            f"at the shared {threshold:.0%} cutoff, the booking is classified **{prediction}**."
        )


def _model_probability(name: str) -> float:
    return float(bundle["models"][name].predict_proba(row)[0, 1])


def _metric_card(label: str, value: str, note: str) -> str:
    return (
        '<div style="background:linear-gradient(145deg,#102a32,#0b1c24);'
        'border:1px solid #24424a;border-radius:14px;padding:14px 16px;min-height:91px">'
        f'<div style="font-size:10px;letter-spacing:.1em;text-transform:uppercase;color:#92abad">{label}</div>'
        f'<div style="font-size:22px;font-weight:800;color:#ecf7f2;margin:5px 0">{value}</div>'
        f'<div style="font-size:10px;color:#7f9a9d">{note}</div></div>'
    )


def _confusion_figure(matrix: np.ndarray, model_name: str) -> go.Figure:
    tn, fp, fn, tp = matrix.ravel()
    labels = [
        [f"TN<br>{tn:,}", f"FP<br>{fp:,}"],
        [f"FN<br>{fn:,}", f"TP<br>{tp:,}"],
    ]
    figure = go.Figure(
        go.Heatmap(
            z=matrix,
            x=["Predicted · stay", "Predicted · cancel"],
            y=["Actual · stay", "Actual · cancel"],
            text=labels,
            texttemplate="%{text}",
            textfont={"size": 15, "color": "#eff8f5"},
            colorscale=[[0, "#10262e"], [0.55, "#126a59"], [1, "#35d49a"]],
            showscale=False,
            hovertemplate="Count: %{z}<extra></extra>",
        )
    )
    figure.update_layout(
        title=f"{model_name} · held-out confusion matrix",
        height=385,
        margin=dict(l=12, r=12, t=55, b=18),
        paper_bgcolor="#0a1c24",
        plot_bgcolor="#0a1c24",
        font=dict(color="#dceae6"),
        yaxis=dict(autorange="reversed"),
    )
    return figure


def _dark_layout(figure: go.Figure, height: int = 420) -> go.Figure:
    figure.update_layout(
        height=height,
        margin=dict(l=22, r=18, t=52, b=35),
        paper_bgcolor="#0a1c24",
        plot_bgcolor="#0a1c24",
        font=dict(color="#dceae6"),
        legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", y=1.12),
        hoverlabel=dict(bgcolor="#102a32", font_color="#eff8f5"),
    )
    figure.update_xaxes(gridcolor="#1b3740", zerolinecolor="#31515a")
    figure.update_yaxes(gridcolor="#1b3740", zerolinecolor="#31515a")
    return figure


if page == "Decision Tree Lab":
    pipeline = bundle["models"]["Decision Tree"]
    tree = pipeline.named_steps["model"]
    probability = _model_probability("Decision Tree")
    depth, node_count, leaf_count = _tree_stats(tree)
    hero(
        "01",
        "Decision Tree Lab",
        "Follow the fitted tree's real questions from the booking root to its final leaf.",
    )
    booking_banner(probability, "Decision Tree")
    m1, m2, m3 = st.columns(3)
    m1.metric("Fitted depth", depth)
    m2.metric("Actual nodes", f"{node_count:,}")
    m3.metric("Terminal leaves", f"{leaf_count:,}")
    visible_depth = st.slider(
        "Visible tree depth",
        min_value=1,
        max_value=min(5, depth),
        value=min(3, depth),
        help="This changes only how many real levels are shown; it does not refit or alter the trained tree.",
    )
    path, _ = trace_tree(pipeline, tree, row)
    nodes = serialize_tree(pipeline, tree, visible_depth, threshold=threshold)
    render_tree(nodes, visible_depth, path, title="Decision Tree · notebook-fitted model")
    _path_expander("Decision Tree", tree, pipeline, probability)
    st.markdown(
        '<div class="note-card">Numeric thresholds are translated back from the notebook scaler to the booking units. '
        'Categorical splits refer to the original one-hot encoded feature. The displayed node counts, impurity, '
        'class mix and path come from the fitted tree.</div>',
        unsafe_allow_html=True,
    )

elif page == "Logistic Regression Lab":
    pipeline = bundle["models"]["Logistic Regression"]
    classifier = pipeline.named_steps["model"]
    probability = _model_probability("Logistic Regression")
    hero(
        "02",
        "Logistic Regression Lab",
        "See the fitted model turn a weighted booking profile into a cancellation probability.",
    )
    booking_banner(probability, "Logistic Regression")
    model_features = [
        "lead_time",
        "adr",
        "previous_cancellations",
        "total_of_special_requests",
        "arrival_date_day_of_month",
        "days_in_waiting_list",
    ]
    selected_feature = st.selectbox(
        "Feature response · change one booking feature, hold the rest fixed",
        model_features,
        format_func=lambda value: value.replace("_", " ").title(),
    )
    training_values = bundle["X_train"][selected_feature].dropna()
    low, high = float(training_values.min()), float(training_values.max())
    values = np.linspace(low, high, 76)
    curve_rows = []
    for value in values:
        changed = booking.copy()
        if selected_feature in COUNT_FIELDS:
            changed[selected_feature] = int(round(value))
        else:
            changed[selected_feature] = float(value)
        curve_rows.append(booking_frame(changed, bundle["feature_names"]).iloc[0])
    curve_frame = pd.DataFrame(curve_rows, columns=bundle["feature_names"])
    curve_probabilities = pipeline.predict_proba(curve_frame)[:, 1]
    current_x = float(booking[selected_feature])
    transformed_booking = pipeline.named_steps["preprocessor"].transform(row)
    current_z = float(classifier.decision_function(transformed_booking)[0])

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[values[0]],
            y=[curve_probabilities[0]],
            mode="lines",
            line=dict(color="#41dfa4", width=3, shape="spline"),
            fill="tozeroy",
            fillcolor="rgba(40,197,143,.10)",
            name="Actual model response",
            hovertemplate="%{x:.2f} → %{y:.1%}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[current_x],
            y=[probability],
            mode="markers",
            marker=dict(size=14, color="#f4c767", line=dict(width=2, color="#fff4d2")),
            name="This booking",
            hovertemplate="Booking: %{x:.2f}<br>Cancellation probability: %{y:.1%}<extra></extra>",
        )
    )
    frames = []
    for index in range(1, len(values) + 1):
        frames.append(
            go.Frame(
                data=[
                    go.Scatter(
                        x=values[:index],
                        y=curve_probabilities[:index],
                        mode="lines",
                        line=dict(color="#41dfa4", width=3, shape="spline"),
                        fill="tozeroy",
                        fillcolor="rgba(40,197,143,.10)",
                    ),
                    go.Scatter(x=[current_x], y=[probability], mode="markers"),
                ],
                traces=[0, 1],
            )
        )
    fig.frames = frames
    fig.update_layout(
        title=f"Cancellation probability as {selected_feature.replace('_', ' ')} changes",
        xaxis_title=selected_feature.replace("_", " ").title(),
        yaxis_title="Model probability of cancellation",
        yaxis=dict(range=[0, 1], tickformat=".0%"),
        xaxis=dict(range=[low, high]),
        shapes=[
            dict(
                type="line",
                xref="paper",
                x0=0,
                x1=1,
                yref="y",
                y0=threshold,
                y1=threshold,
                line=dict(color="#f3c86b", width=2, dash="dash"),
            )
        ],
        annotations=[
            dict(
                x=0.99,
                y=threshold,
                xref="paper",
                yref="y",
                text=f"cutoff {threshold:.0%}",
                showarrow=False,
                xanchor="right",
                yshift=10,
                font=dict(color="#f3c86b", size=11),
            )
        ],
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                x=0,
                y=1.18,
                buttons=[
                    dict(
                        label="▶ Draw curve",
                        method="animate",
                        args=[
                            None,
                            {
                                "frame": {"duration": 28, "redraw": True},
                                "fromcurrent": True,
                                "transition": {"duration": 0},
                            },
                        ],
                    ),
                    dict(
                        label="↺ Restart",
                        method="animate",
                        args=[
                            [None],
                            {
                                "frame": {"duration": 0, "redraw": True},
                                "mode": "immediate",
                                "transition": {"duration": 0},
                            },
                        ],
                    ),
                ],
            )
        ],
    )
    _dark_layout(fig, 500)
    st.plotly_chart(fig, width="stretch")

    left, right = st.columns([1.25, 0.75])
    with left:
        st.subheader("What the model is calculating")
        st.latex(r"P(\mathrm{cancel}) = \frac{1}{1 + e^{-z}}")
        st.markdown(
            f'<div class="note-card">For this booking the fitted model score is '
            f'<b>z = {current_z:.3f}</b>, giving <b>{probability:.1%}</b> cancellation probability. '
            f'The gold line is the shared classification cutoff ({threshold:.0%}); moving it changes '
            f'the class, not the probability.</div>',
            unsafe_allow_html=True,
        )
    with right:
        predicted = "CANCEL" if probability >= threshold else "KEEP"
        st.markdown("#### Current booking outcome")
        st.markdown(
            _metric_card("Probability", f"{probability:.1%}", f"Cutoff {threshold:.0%}")
            + _metric_card("Classification", predicted, "Based on the shared cutoff"),
            unsafe_allow_html=True,
        )

    dense = (
        transformed_booking.toarray()[0]
        if hasattr(transformed_booking, "toarray")
        else np.asarray(transformed_booking[0])
    )
    names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    contributions = dense * classifier.coef_[0]
    order = np.argsort(np.abs(contributions))[-12:]
    order = order[np.argsort(contributions[order])]
    impact = go.Figure(
        go.Bar(
            x=contributions[order],
            y=[str(names[i]).replace("_", " ") for i in order],
            customdata=classifier.coef_[0][order],
            orientation="h",
            marker_color=[
                "#ef9f70" if contributions[i] > 0 else "#35c995" for i in order
            ],
            hovertemplate=(
                "%{y}<br>Actual fitted coefficient: %{customdata:.3f}"
                "<br>Contribution to log-odds: %{x:.3f}<extra></extra>"
            ),
        )
    )
    impact.update_layout(
        title="Largest fitted coefficient contributions for this booking",
        xaxis_title="Contribution to model score z",
        yaxis_title="",
    )
    _dark_layout(impact, 430)
    st.plotly_chart(impact, width="stretch")
    st.caption(
        "This feature-response plot uses actual pipeline predictions, varying only the selected booking field. "
        "Lead-time bands are recomputed from the notebook's original bins, so that curve can contain real band transitions."
    )

elif page == "Random Forest Lab":
    pipeline = bundle["models"]["Random Forest"]
    forest = pipeline.named_steps["model"]
    probability = _model_probability("Random Forest")
    hero(
        "03",
        "Random Forest Lab",
        "Watch multiple fitted trees contribute probability estimates to one ensemble decision.",
    )
    booking_banner(probability, "Random Forest")
    transformed = pipeline.named_steps["preprocessor"].transform(row)
    tree_probabilities = np.array(
        [float(estimator.predict_proba(transformed)[0, 1]) for estimator in forest.estimators_]
    )
    forest_average = float(tree_probabilities.mean())
    st.markdown(
        '<div class="note-card">Aggregation is the Random Forest\'s actual probability average: '
        f'<b>{len(forest.estimators_)} trees</b> contribute equally. The forest probability is '
        f'<b>{forest_average:.1%}</b>; the shared cutoff labels it '
        f'<b>{"cancel" if forest_average >= threshold else "not cancel"}</b>. '
        'The cards below animate eight actual members; the aggregate uses all fitted members.</div>',
        unsafe_allow_html=True,
    )
    sample_indexes = np.linspace(
        0, len(forest.estimators_) - 1, num=min(8, len(forest.estimators_)), dtype=int
    )
    cards: list[dict[str, Any]] = []
    for index in sample_indexes:
        estimator = forest.estimators_[int(index)]
        root = serialize_tree(
            pipeline, estimator, max_depth=1, threshold=threshold
        )[0]
        cards.append(
            {
                "index": int(index),
                "root": root["condition"],
                "probability": float(tree_probabilities[int(index)]),
                "threshold": threshold,
                "samples": root["samples"],
                "forest_probability": forest_average,
                "forest_size": len(forest.estimators_),
            }
        )
    render_forest(cards)

    options = list(range(len(forest.estimators_)))
    selected_index = st.selectbox(
        "Explore a fitted tree",
        options,
        index=0,
        format_func=lambda value: f"Tree {value + 1} of {len(forest.estimators_)}",
    )
    selected_tree = forest.estimators_[selected_index]
    selected_probability = float(tree_probabilities[selected_index])
    selected_max_depth = max(1, min(5, int(selected_tree.get_depth())))
    selected_depth = st.slider(
        "Selected tree visualization depth",
        min_value=1,
        max_value=selected_max_depth,
        value=min(3, selected_max_depth),
        key="forest_tree_depth",
        help="Reveal additional levels from this fitted member; the tree is not refitted.",
    )
    selected_path, _ = trace_tree(pipeline, selected_tree, row)
    st.markdown(
        f'<div class="note-card">Tree {selected_index + 1} predicts '
        f'<b>{selected_probability:.1%}</b> cancellation probability for this booking. '
        f'Its individual output is not the final forest score; it is one term in the mean of '
        f'{len(forest.estimators_)} tree probabilities.</div>',
        unsafe_allow_html=True,
    )
    render_tree(
        serialize_tree(
            pipeline, selected_tree, selected_depth, threshold=threshold
        ),
        selected_depth,
        selected_path,
        height=720,
        title=f"Random Forest · actual tree {selected_index + 1}",
    )
    _path_expander("Random Forest tree", selected_tree, pipeline, selected_probability)

    left, right = st.columns([1.2, 0.8])
    with left:
        distribution = go.Figure(
            go.Histogram(
                x=tree_probabilities,
                nbinsx=20,
                marker_color="#24bd88",
                opacity=0.88,
                name="Per-tree probability",
                hovertemplate="Cancellation probability: %{x:.1%}<br>Trees: %{y}<extra></extra>",
            )
        )
        distribution.add_vline(
            x=forest_average,
            line_color="#f0c66e",
            line_dash="dash",
            line_width=3,
            annotation_text=f"mean {forest_average:.1%}",
            annotation_font_color="#f0c66e",
        )
        distribution.update_layout(
            title="Actual per-tree probability distribution · all trees",
            xaxis_title="Tree cancellation probability",
            yaxis_title="Number of trees",
            xaxis=dict(range=[0, 1], tickformat=".0%"),
        )
        _dark_layout(distribution, 390)
        st.plotly_chart(distribution, width="stretch")
    with right:
        st.markdown("#### Probability aggregation")
        st.markdown(
            _metric_card("Mean of tree probabilities", f"{forest_average:.1%}", f"1 / {len(tree_probabilities)} × sum of all outputs")
            + _metric_card("Shared cutoff", f"{threshold:.0%}", f"Final class: {'CANCEL' if forest_average >= threshold else 'KEEP'}"),
            unsafe_allow_html=True,
        )

elif page == "Model Comparison Lab":
    hero(
        "04",
        "Model Comparison Lab",
        "Compare all three fitted pipelines on the same held-out bookings and move the shared cutoff live.",
    )
    st.markdown(
        f'<div class="note-card">Test set: <b>{bundle["test_rows"]:,} bookings</b>, '
        f'stratified from the notebook workflow. Threshold-dependent scores below use the current '
        f'<b>{threshold:.0%}</b> cutoff. ROC-AUC is computed from probabilities and does not change with the cutoff.</div>',
        unsafe_allow_html=True,
    )
    metric_names = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    columns = st.columns(3)
    for column, (name, values) in zip(columns, metrics.items()):
        with column:
            st.markdown(f"#### {name}")
            for metric_name in metric_names:
                st.markdown(
                    _metric_card(
                        metric_name,
                        f"{values[metric_name]:.3f}",
                        "cutoff-based" if metric_name != "ROC-AUC" else "threshold-independent",
                    ),
                    unsafe_allow_html=True,
                )

    st.subheader("Metric comparison")
    comparison = go.Figure()
    for name, result in metrics.items():
        comparison.add_trace(
            go.Bar(
                name=name,
                x=metric_names,
                y=[result[key] for key in metric_names],
                hovertemplate=f"{name}<br>%{{x}}: %{{y:.3f}}<extra></extra>",
            )
        )
    comparison.update_layout(
        barmode="group",
        yaxis=dict(range=[0, 1], tickformat=".0%"),
        title=f"Test metrics at a {threshold:.0%} cutoff",
        yaxis_title="Score",
    )
    _dark_layout(comparison, 410)
    st.plotly_chart(comparison, width="stretch")

    left, right = st.columns([1.05, 0.95])
    with left:
        roc_figure = go.Figure()
        for name, (false_positive, true_positive, _) in bundle["roc"].items():
            auc = metrics[name]["ROC-AUC"]
            roc_figure.add_trace(
                go.Scatter(
                    x=false_positive,
                    y=true_positive,
                    mode="lines",
                    name=f"{name} · AUC {auc:.3f}",
                    line=dict(width=2.5),
                )
            )
        roc_figure.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                name="Random baseline",
                line=dict(color="#819496", dash="dash"),
            )
        )
        roc_figure.update_layout(
            title="ROC curves · held-out test set",
            xaxis_title="False positive rate",
            yaxis_title="True positive rate",
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1]),
        )
        _dark_layout(roc_figure, 455)
        st.plotly_chart(roc_figure, width="stretch")
    with right:
        selected_model = st.selectbox("Confusion matrix", list(metrics))
        st.plotly_chart(
            _confusion_figure(metrics[selected_model]["Confusion"], selected_model),
            width="stretch",
        )
        st.caption("Rows are actual outcomes; columns are model predictions. TN/FP/FN/TP are labeled in each cell.")

    st.subheader("What the measured trade-offs mean")
    best_auc = max(metrics, key=lambda name: metrics[name]["ROC-AUC"])
    best_recall = max(metrics, key=lambda name: metrics[name]["Recall"])
    best_precision = max(metrics, key=lambda name: metrics[name]["Precision"])
    st.markdown(
        f'<div class="insight">On this held-out test set, <b>{best_auc}</b> has the highest ROC-AUC '
        f'({metrics[best_auc]["ROC-AUC"]:.3f}); <b>{best_recall}</b> has the highest cancellation recall '
        f'({metrics[best_recall]["Recall"]:.3f}) and <b>{best_precision}</b> has the highest cancellation precision '
        f'({metrics[best_precision]["Precision"]:.3f}) at the current cutoff. '
        '<b>Decision Tree</b> is the most directly interpretable through explicit split rules; '
        'Logistic Regression exposes directional encoded-feature contributions, while the Random Forest '
        'trades simple global explanations for an ensemble score. For hotel operations, prioritize recall '
        'when missed cancellations are costly, and precision when retention outreach is expensive. '
        'No model is best for every cost profile.</div>',
        unsafe_allow_html=True,
    )
    st.markdown("#### Notebook-faithful pipeline")
    with st.expander("Data, features, preprocessing and exact model settings"):
        st.markdown(
            f"""
- Source: the bundled `hotel_bookings.csv` contained in `data/hotel_bookings.zip` ({len(bundle["df"]):,} rows after the notebook's duplicate, target and domain-validity cleaning).
- Target: `is_canceled`; reservation outcome/status date and assigned room, agent, company are excluded to avoid leakage/identifiers.
- Engineered: total nights, total guests, numeric arrival month, four-season grouping and the notebook's five lead-time bands.
- Split: stratified 80/20, random state 42 (train {bundle["train_rows"]:,}; test {bundle["test_rows"]:,}).
- Numeric preprocessing: training-split skew chooses mean vs median imputation (absolute skew > 1 uses median), then StandardScaler. Categorical preprocessing: most-frequent imputation and OneHotEncoder(handle_unknown="ignore").
- Logistic Regression: `max_iter=2000`, `class_weight=None`, `random_state=42`.
- Decision Tree: `max_depth=12`, `min_samples_leaf=20`, `random_state=42`.
- Random Forest: `n_estimators=150`, `max_depth=18`, `min_samples_leaf=5`, `n_jobs=-1`, `random_state=42`.
"""
        )

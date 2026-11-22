# AURELIA · Hotel Cancellation Model Lab

A Streamlit learning lab that turns the supplied hotel-cancellation notebook into four live, interactive model demonstrations:

1. **Decision Tree Lab** — progressively reveals actual fitted nodes, explains node statistics and traces a booking through the full tree.
2. **Logistic Regression Lab** — animates actual pipeline probabilities as one booking feature changes, displays the shared cutoff and per-feature log-odds contributions.
3. **Random Forest Lab** — progressively reveals actual forest members, explores a selected fitted tree and shows the probability average across all 150 trees.
4. **Model Comparison Lab** — compares metrics, ROC curves and confusion matrices on the notebook's held-out test data.

The shared booking editor and cancellation threshold are in the sidebar. The app retrains and caches the three notebook-specified pipelines on first launch using the self-contained compressed CSV in `data/hotel_bookings.zip`. It does not depend on files on the author's computer, download data at runtime, or use sample/fabricated predictions.

## Run locally

Use Python 3.11 (also suitable for Streamlit Community Cloud):

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

The first page load fits the models and may take a little time. Streamlit caches the fitted pipelines for the active app process; no model binaries need to be committed.

## Deploy on Streamlit Community Cloud

1. Upload the contents of this folder to a GitHub repository (or push this folder as its own repository).
2. In [Streamlit Community Cloud](https://share.streamlit.io/), create an app from that repository.
3. Select `app.py` as the **Main file path**.
4. Deploy. The app installs `requirements.txt` and reads the bundled dataset using a relative path.

No secrets, local paths, private services, or external data downloads are required. The embedded SVG controls and Plotly animations run in the hosted browser.

## Notebook methodology retained

`modeling.py` follows the supplied notebook's workflow:

- Read `hotel_bookings.csv` from the bundled ZIP; remove exact duplicate rows and invalid targets; coerce `children` to numeric. As in the notebook's domain checks, remove only negative lead-time, ADR or stay values (the notebook finds one negative-ADR row); do not trim plausible IQR outliers.
- Engineer `total_nights`, `total_guests`, numeric arrival month, the notebook's four seasons and five lead-time bands.
- Exclude `reservation_status`, `reservation_status_date`, `assigned_room_type`, `agent`, `company`, and the replaced `arrival_date_month` field from predictors.
- Use a stratified 80/20 split with `random_state=42`.
- Use training-split skew to choose mean vs median numeric imputation (absolute skew greater than 1 uses median), followed by `StandardScaler`; impute categories by most-frequent and one-hot encode with `handle_unknown="ignore"`.
- Train Logistic Regression (`max_iter=2000`, `class_weight=None`, seed 42), Decision Tree (`max_depth=12`, `min_samples_leaf=20`, seed 42), and Random Forest (`150` trees, `max_depth=18`, `min_samples_leaf=5`, `n_jobs=-1`, seed 42).

Threshold changes only turn probabilities into labels; they do not refit models or change ROC-AUC. The app computes the scores and visuals from the actual fitted pipelines and held-out predictions.

### Notebook's recorded 0.50-cutoff reference scores

The notebook reports, on its recorded test split:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.7998 | 0.6836 | 0.5055 | 0.5812 | 0.8478 |
| Decision Tree | 0.8161 | 0.6868 | 0.6087 | 0.6454 | 0.8698 |
| Random Forest | 0.8168 | 0.7751 | 0.4697 | 0.5849 | 0.8785 |

These are notebook-recorded reference outputs, not hardcoded app results. The live app recomputes its metrics from the bundled dataset and its fitted split; small differences can occur if the source file or scikit-learn version differs from the notebook runtime.

## Project files

- `app.py` — Streamlit interface and four labs.
- `modeling.py` — original-data preparation, feature engineering, training and tree inspection helpers.
- `components.py` — self-contained animated SVG tree and forest components.
- `data/hotel_bookings.zip` — original CSV bundled for deployment.
- `tests/test_modeling.py` — focused tests for booking feature engineering and threshold behavior.
- `TESTING.md` — verification record.

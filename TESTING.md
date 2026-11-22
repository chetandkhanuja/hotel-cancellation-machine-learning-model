# Testing and verification

Verification performed locally on Windows with Python 3.14, Streamlit 1.65.0 and scikit-learn 1.7.2. Notebook reference values are documented separately in `README.md`.

## Checks

- [x] Focused unit tests (`python -m pytest -q`) — **4 passed**
- [x] Dataset load and notebook-faithful preprocessing — **87,395 cleaned rows; stratified split 69,916 train / 17,479 test**
- [x] Fit all three models — completed in about 18 seconds locally; 236 transformed features and all three models produced held-out probabilities.
- [x] Streamlit application startup — `/_stcore/health` returned `ok`.
- [x] Streamlit AppTest — all four sections rendered with no exceptions at the default and a non-default shared threshold.
- [x] Browser checks — after a fresh server restart, tree nodes rendered with explicit SVG coordinates; selecting the root showed its split and child statistics, and tracing the booking reached leaf 664. The forest's eight displayed members, selected-tree visualization and 150-tree aggregate rendered; Logistic Regression curve control and comparison metrics rendered.
- [ ] Streamlit Community Cloud deployment

## Recomputed held-out metrics

Computed locally from the packaged data at a 0.50 threshold:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.7995 | 0.6829 | 0.5051 | 0.5807 | 0.8478 |
| Decision Tree | 0.8164 | 0.6859 | 0.6123 | 0.6470 | 0.8698 |
| Random Forest | 0.8168 | 0.7751 | 0.4697 | 0.5849 | 0.8785 |

## Limitations

The notebook-recorded Logistic Regression and Decision Tree threshold metrics differ slightly from the local scikit-learn 1.7.2 refit; their ROC-AUC and all Random Forest reference metrics match. The application always displays values calculated from its own fitted pipelines and held-out predictions. A live Streamlit Community Cloud deployment has not been performed.

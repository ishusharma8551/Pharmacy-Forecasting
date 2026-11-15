"""Page 5 of 6: Model Performance."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import streamlit as st  # noqa: E402

from src import ui  # noqa: E402

ui.setup("Model Performance", "🧪")
st.warning("All results below were produced by running the code on SYNTHETIC data. They demonstrate that the pipeline "
           "works; they are not evidence of real-world accuracy.")
ui.get_data()
dm, em = ui.load_report("demand_model_comparison.csv"), ui.load_report("expiry_model_comparison.csv")
if dm is None or em is None:
    st.error("Reports not found. Run `python train.py` first.")
    st.stop()

tab1, tab2 = st.tabs(["Demand forecasting", "Expiry-risk classification"])
with tab1:
    test = dm[dm["Split"] == "test"]
    st.markdown("**Protocol:** chronological split. The last 90 days are the test set; models are chosen on a validation period "
                "before it. Baselines need no training - ML models must beat them to be worthwhile.")
    metric = st.radio("Metric", ["MAE", "RMSE", "MAPE_%", "WAPE_%"], horizontal=True)
    fig = px.bar(test, x="Horizon", y=metric, color="Model", barmode="group", title=f"Test {metric} by horizon (lower is better)")
    ui.chart(fig)
    ui.table(test.sort_values(["Horizon", "MAE"]).round(2))
    sel = dm[(dm["Selected"]) & (dm["Split"] == "validation")][["Horizon", "Model"]]
    st.write("**Model chosen on validation, per horizon:**", ", ".join(f"{r.Horizon}d -> {r.Model}" for r in sel.itertuples()))
    best_test = test.sort_values("MAE").groupby("Horizon").first()["Model"]
    st.write("**Lowest test MAE, per horizon:**", ", ".join(f"{h}d -> {m}" for h, m in best_test.items()))
    st.caption("MAPE ignores days where the actual demand is zero. WAPE = total absolute error / total actual demand.")
    imp = ui.load_report("demand_feature_importance.csv")
    if imp is not None:
        h = st.selectbox("Feature importance - horizon (days)", sorted(imp["Horizon"].unique()))
        i = imp[imp["Horizon"] == h].sort_values("Importance")
        ui.chart(px.bar(i, x="Importance", y="Feature", orientation="h", title=f"{h}-day model: {i['Model'].iloc[0]}"))
    pred = ui.load_report("forecast_test_predictions.csv")
    if pred is not None:
        h2 = st.selectbox("Forecast performance - horizon (days)", sorted(pred["Horizon"].unique()), key="h2")
        p = pred[pred["Horizon"] == h2]
        fig = px.scatter(p.sample(min(len(p), 1500), random_state=0), x="Actual", y="Predicted", opacity=0.4,
                         title=f"Predicted vs actual units over {h2} days (test set sample)")
        mx = float(max(p["Actual"].max(), p["Predicted"].max()))
        fig.add_shape(type="line", x0=0, y0=0, x1=mx, y1=mx, line=dict(color="red", dash="dash"))
        ui.chart(fig)

with tab2:
    t = em[em["Split"] == "test"]
    st.markdown("**Labels** come from what actually happened to a batch (unsold share at expiry): "
                "Low < 15%, Medium 15-50%, High >= 50%. Selection prioritises **recall on High risk**.")
    ui.table(t.drop(columns=["N"]).round(3))
    melt = t.melt(id_vars="Model", value_vars=["Accuracy", "F1_macro", "Recall_High"], var_name="Metric", value_name="Score")
    ui.chart(px.bar(melt, x="Metric", y="Score", color="Model", barmode="group", title="Test metrics"))
    rule = t[t["Model"].str.startswith("Rule")].iloc[0]
    ml = t[~t["Model"].str.startswith("Rule")].sort_values("F1_macro", ascending=False).iloc[0]
    st.info(f"Honest comparison: the simple rule baseline (thresholds on the forecast unsold ratio) scores macro-F1 "
            f"{rule['F1_macro']:.3f}; the best ML model scores {ml['F1_macro']:.3f} ({ml['Model']}). "
            "When these are close, the rule is a perfectly defensible, more explainable choice.")
    cm = ui.load_report("expiry_confusion_matrices.json")
    if cm:
        name = st.selectbox("Confusion matrix - model", list(cm["matrices"].keys()), index=1)
        mat = pd.DataFrame(cm["matrices"][name], index=[f"Actual {l}" for l in cm["labels"]], columns=[f"Pred {l}" for l in cm["labels"]])
        ui.chart(px.imshow(mat, text_auto=True, color_continuous_scale="Blues", title=f"Confusion matrix (test): {name}"))
        st.caption(f"Test label distribution: {cm['label_distribution']['test']}")
    ei = ui.load_report("expiry_feature_importance.csv")
    if ei is not None:
        ui.chart(px.bar(ei.sort_values("Importance"), x="Importance", y="Feature", orientation="h",
                        title=f"Feature importance - {ei['Model'].iloc[0]}"))
st.info(ui.EXPIRY_DISCLAIMER)
ui.footer()

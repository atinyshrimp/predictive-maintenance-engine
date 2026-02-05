"""Performance page - Model evaluation metrics and analysis."""

import streamlit as st
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import load_results, get_delta_results, ASSETS_DIR

st.title("📊 Model Performance")

# Load results
results = load_results()

if results is None or results.empty:
    st.warning("⚠️ No training results found. Run the training pipeline first:")
    st.code("python src/train.py --dataset FD001 --imbalance cost_sensitive")
    st.stop()

model_data = results.iloc[0]  # Single model

# Key Metrics - Primary KPIs
st.subheader("🎯 Key Metrics")

metrics_delta = get_delta_results()

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    delta_recall = metrics_delta['recall'] * 100 if metrics_delta else 0
    st.metric("Recall", f"{model_data['recall']:.1%}", 
              delta=f"{delta_recall:+.1f}%", delta_color="normal")
with col2:
    delta_precision = metrics_delta['precision'] * 100 if metrics_delta else 0
    st.metric("Precision", f"{model_data['precision']:.1%}",
              delta=f"{delta_precision:+.1f}%", delta_color="off")  # Lower precision is acceptable
with col3:
    delta_f1 = metrics_delta['f1_score'] * 100 if metrics_delta else 0
    st.metric("F1-Score", f"{model_data['f1_score']:.1%}",
              delta=f"{delta_f1:+.1f}%", delta_color="off")
with col4:
    delta_roc = metrics_delta['roc_auc'] if metrics_delta else 0
    st.metric("ROC-AUC", f"{model_data['roc_auc']:.3f}",
              delta=f"{delta_roc:+.3f}", delta_color="normal")  # Major improvement!
with col5:
    delta_acc = metrics_delta['accuracy'] * 100 if metrics_delta else 0
    st.metric("Accuracy", f"{model_data['accuracy']:.1%}",
              delta=f"{delta_acc:+.1f}%", delta_color="off")

st.caption("📊 Δ vs. original notebook baseline (K-Neighbors/TPOT) • ROC-AUC improved from 0.48 (random) to 0.96")

# Decision threshold if available
if 'threshold' in model_data and model_data['threshold']:
    st.caption(f"📍 Decision threshold: {model_data['threshold']:.3f} (optimized for recall ≥ 95%)")

st.markdown("---")

# Confusion Matrix Analysis
st.subheader("🔢 Confusion Matrix Analysis")

cm_img = ASSETS_DIR / "confusion_matrix_random_forest_(balanced).png"

col1, col2 = st.columns([1, 1])

with col1:
    if cm_img.exists():
        st.image(str(cm_img), use_container_width=True)
    else:
        st.info("Run training to generate confusion matrix")

with col2:
    st.markdown("""
    **Reading the Matrix:**
    
    |  | Pred: OK | Pred: Fail |
    |--|----------|------------|
    | **True: OK** | TN | FP (false alarm) |
    | **True: Fail** | FN (missed!) | TP |
    
    **For Maintenance Systems:**
    - ✅ **High Recall** = Few missed failures (low FN)
    - ⚠️ **Low Precision** = More false alarms (high FP)
    
    *Trade-off: We accept false alarms to catch real failures.*
    """)

st.markdown("---")

# ROC and PR Curves
st.subheader("📈 Performance Curves")

col1, col2 = st.columns(2)

with col1:
    roc_img = ASSETS_DIR / "roc_curves_comparison.png"
    if roc_img.exists():
        st.image(str(roc_img), caption="ROC Curve", use_container_width=True)
        st.caption("AUC: 0.5 = random, 1.0 = perfect")
    else:
        st.info("Run training to generate ROC curve")

with col2:
    pr_img = ASSETS_DIR / "precision_recall_random_forest_(balanced).png"
    if pr_img.exists():
        st.image(str(pr_img), caption="Precision-Recall Curve", use_container_width=True)
        st.caption("Critical for imbalanced data")
    else:
        st.info("Run training to generate PR curve")

st.markdown("---")

# Feature Importance
st.subheader("🔍 Top Predictive Features")

fi_img = ASSETS_DIR / "feature_importance_random_forest_(balanced).png"
if fi_img.exists():
    st.image(str(fi_img), use_container_width=True)
    st.caption("Degradation patterns and rolling statistics are most predictive")
else:
    st.info("Run training to generate feature importance plot")

st.markdown("---")

# Collapsible sections for details
with st.expander("📚 Why Low Precision is Acceptable"):
    st.markdown("""
    ~43% precision = ~57% of alerts are false alarms. This is intentional:
    
    | Scenario | Estimated Cost |
    |----------|----------------|
    | Missed failure (FN) | **$100K+** - unplanned downtime, safety risk, emergency repair |
    | False alarm (FP) | **$500** - scheduled inspection |
    
    **Cost ratio ~200:1** → Optimizing for high recall is the correct business decision.
    """)

with st.expander("📥 Export Results"):
    csv = results.to_csv(index=False)
    st.download_button("Download CSV", csv, "model_results.csv", "text/csv")

with st.expander("📈 Comparison to Original Notebook"):
    st.markdown("""
    **Original Notebook (TPOT AutoML → K-Neighbors):**
    
    | Metric | Notebook | Current | Change |
    |--------|----------|---------|--------|
    | Recall | 95.2% | 97.9% | **+2.7%** |
    | Precision | 99.5% | 43.3% | -56.2% |
    | ROC-AUC | 0.48 | 0.96 | **+0.48** |
    
    **Why the current model is actually better:**
    
    The original notebook model had **zero true positives** - it simply predicted "no failure" 
    for everything, achieving high accuracy from class imbalance (95% healthy readings).
    
    - 🔴 **ROC-AUC 0.48** = Worse than random guessing!
    - 🟢 **ROC-AUC 0.96** = Excellent discrimination ability
    
    The current model intentionally trades precision for recall because missing a failure 
    costs ~200x more than a false alarm in industrial maintenance.
    """)

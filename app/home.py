"""Home page for Predictive Maintenance Engine."""

import streamlit as st
import requests
from utils import load_model, load_results

st.title("🔧 Predictive Maintenance Engine")
st.markdown("### AI-Powered Equipment Failure Prediction")

st.markdown("""
Welcome to the **Predictive Maintenance Engine** - a machine learning system 
for predicting industrial equipment failures using the NASA Turbofan dataset.
""")

# Status indicators
col1, col2, col3 = st.columns(3)

with col1:
    with st.spinner("Loading model..."):
        model = load_model()
    if model:
        st.success("✅ Model Ready")
    else:
        st.warning("⚠️ No model")

with col2:
    results = load_results()
    if results is not None:
        recall = results['recall'].max()
        st.success(f"🎯 {recall:.0%} Recall")
    else:
        st.info("📊 Not trained")
        
with col3:
    with st.spinner("Checking API..."):
        try:
            response = requests.get("http://localhost:8000/health")
            st.session_state.is_api_running = response.status_code == 200
        except requests.RequestException:
            st.session_state.is_api_running = False

    if st.session_state.get("is_api_running"):
        st.success("🚀 API Running")
    else:        
        st.warning("⚠️ API Offline")

st.markdown("---")

# Quick navigation
st.subheader("📍 Navigation")

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("""
    **🔮 Predictions**

    Interactive failure prediction with timeline analysis
    """)
with col2:
    st.markdown("""
    **📊 Performance**

    Model metrics, confusion matrix, ROC curves
    """)
with col3:
    st.markdown(""" 
    **ℹ️ About**
    
    Technical details and documentation
    """)  # noqa: RUF001

st.markdown("---")

# Quick Start
st.subheader("🚀 Quick Start")

st.code("""
# 1. Train the model
python src/train.py

# 2. Start the API
cd api && python app.py

# 3. Use this dashboard or call API at http://localhost:8000/predict
""", language="bash")

st.markdown("---")

# Architecture overview
st.subheader("🏗️ Pipeline")

st.markdown("""
```
Sensor Data → Feature Engineering → Random Forest → Risk Assessment
   (26 sensors)    (137 features)     (balanced)      (LOW/MED/HIGH)
```
""")

# Footer
st.markdown("---")
st.caption("Built with Streamlit, FastAPI, and scikit-learn | NASA C-MAPSS Dataset")

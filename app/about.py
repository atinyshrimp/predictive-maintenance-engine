"""About page - Project documentation."""

import streamlit as st

st.title("ℹ️ About This Project") # noqa: RUF001

st.markdown("""
## Predictive Maintenance Engine

A machine learning system for predicting industrial equipment failures 
using the NASA Turbofan Jet Engine dataset.

---

### 🎯 Problem Statement

Industrial equipment failures cause:
- **Unplanned downtime**: Millions in lost productivity
- **Safety risks**: Potential injuries and environmental damage
- **Repair costs**: Emergency repairs cost 3-10x more than planned maintenance

**Solution**: Predict failures before they occur → proactive maintenance.

---

### 📊 Dataset: NASA Turbofan (C-MAPSS)

| Property | Details |
|----------|---------|
| **Source** | NASA Prognostics Center of Excellence |
| **Type** | Run-to-failure simulation |
| **Engines** | 100 units (FD001) |
| **Features** | 21 sensors + 3 operational settings |
| **Target** | Binary failure prediction (RUL < threshold) |

---

### 🔧 Technical Approach

**Feature Engineering** (26 raw → 137 features)
- Rolling statistics: mean, std, EMA (windows: 3, 5)
- Degradation features: cycle position, rate of change, cumulative sum

**Model**
- Random Forest with balanced class weights
- Threshold optimized for ≥95% recall

**Imbalance Handling**
- Balanced class weights (cost-sensitive)
- Also supports: SMOTE, Random Undersampling

---

### 🛠️ Tech Stack

| Layer | Technology |
|-------|------------|
| **ML** | scikit-learn, pandas, NumPy |
| **API** | FastAPI, Pydantic, uvicorn |
| **UI** | Streamlit, Plotly |

---

### 📚 References

- [NASA C-MAPSS Dataset](https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository)
- A. Saxena, K. Goebel, D. Simon, and N. Eklund, “Damage Propagation Modeling
for Aircraft Engine Run-to-Failure Simulation.” https://data.nasa.gov/Aerospace/CMAPSS-Jet-Engine-Simulated-Data/ff5v-kuh6/about_data, oct 2008. Accessed: 2024-
12-31

---

### 👤 Author

[GitHub](https://github.com/atinyshrimp) · [LinkedIn](https://linkedin.com/in/joyce-lapilus)
""")

"""
Predictive Maintenance Engine - Main Entry Point
Multi-page Streamlit application
"""

import streamlit as st
from pathlib import Path
import sys

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils import get_delta_results, load_results

# Page configuration
st.set_page_config(
    page_title="Predictive Maintenance Engine",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Setup navigation with all pages
pg = st.navigation([
    st.Page("home.py", title="Home", icon="🏠", default=True),
    st.Page("predictions.py", title="Predictions", icon="🔮"),
    st.Page("performance.py", title="Performance", icon="📊"),
    st.Page("about.py", title="About", icon="ℹ️"), # noqa: RUF001
])

# Add sidebar info
with st.sidebar:
    st.markdown("### 🔧 Predictive Maintenance")
    st.caption("AI-powered failure prediction")
    
    st.markdown("---")
    
    results = load_results()
    metrics_delta = get_delta_results()
    
    # Model Performance Summary
    st.markdown("**📊 Model Performance**")
    
    if results is not None and not results.empty:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Recall", f"{results.iloc[0]['recall']:.1%}", 
                    delta=f"{metrics_delta['recall'] * 100:+.1f}%" if metrics_delta else "N/A", 
                    delta_color="normal")
        with col2:
            st.metric("ROC-AUC", f"{results.iloc[0]['roc_auc']:.3f}", 
                    delta=f"{metrics_delta['roc_auc']:+.3f}" if metrics_delta else "N/A", 
                    delta_color="normal")
        st.caption("vs. original notebook baseline")
    else:
        st.info("📊 No results found. Train the model to see performance metrics.")
    
    st.markdown("---")
    
    # Dataset Info
    st.markdown("**📁 Dataset**")
    st.markdown("""
    NASA Turbofan (FD001)  
    `100 engines • 26 sensors`  
    `17,731 training cycles`
    """)
    
    st.markdown("---")
    
    # Tech Stack
    st.markdown("**🛠️ Tech Stack**")
    st.markdown("""
    `Python` `scikit-learn` `FastAPI`  
    `Streamlit` `Pandas` `NumPy`
    """)
    
    st.markdown("---")
    
    # Links
    st.markdown("**🔗 Links**")
    st.markdown(f"""
    {"- [API Docs](http://localhost:8000/docs)" if st.session_state.get("is_api_running") else "- API Docs (start API to access)"}
    - [GitHub](https://github.com/atinyshrimp/predictive-maintenance-engine)
    """)
    
    # Footer
    st.markdown("---")
    st.caption("Built by Joyce Lapilus • 2025")

pg.run()
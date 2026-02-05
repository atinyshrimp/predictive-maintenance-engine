"""
Predictive Maintenance Engine - Main Entry Point
Multi-page Streamlit application
"""

import streamlit as st
from pathlib import Path
import sys

# Page configuration
st.set_page_config(
    page_title="Predictive Maintenance Engine",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Setup navigation with all pages
pg = st.navigation([
    st.Page("home.py", title="Home", icon="🏠", default=True),
    st.Page("predictions.py", title="Predictions", icon="🔮"),
    st.Page("performance.py", title="Performance", icon="📊"),
    st.Page("about.py", title="About", icon="ℹ️"),
])

# Add sidebar info
with st.sidebar:
    st.markdown("### 🔧 Predictive Maintenance")
    st.caption("AI-powered failure prediction")
    st.markdown("---")
    st.markdown("""
    **Quick Links:**
    - [API Docs](http://localhost:8000/docs)
    - [GitHub](https://github.com/atinyshrimp/predictive-maintenance-engine)
    """)

pg.run()
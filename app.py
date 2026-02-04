"""Streamlit web application for Predictive Maintenance Engine."""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px

from src.data_loader import TurbofanDataLoader
from src.feature_engineering import FeatureEngineer
from src.config import MODEL_CONFIG

# Page configuration
st.set_page_config(
    page_title="Predictive Maintenance Engine",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .risk-low {
        color: #28a745;
        font-weight: bold;
    }
    .risk-medium {
        color: #ffc107;
        font-weight: bold;
    }
    .risk-high {
        color: #fd7e14;
        font-weight: bold;
    }
    .risk-critical {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    """Load the trained model."""
    model_path = Path("models/best_pipeline.pkl")
    if model_path.exists():
        return joblib.load(model_path)
    return None


@st.cache_data
def load_sample_data(dataset_name="FD001"):
    """Load sample dataset for demonstration."""
    try:
        loader = TurbofanDataLoader(dataset_name=dataset_name)
        train_df, test_df = loader.prepare_data(include_test_rul=True)
        
        # Apply feature engineering
        feature_engineer = FeatureEngineer()
        train_df = feature_engineer.engineer_all_features(train_df, include_rolling=True)
        test_df = feature_engineer.engineer_all_features(test_df, include_rolling=True)
        
        return train_df, test_df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None


def get_risk_level(probability):
    """Determine risk level from probability."""
    if probability < 0.3:
        return "LOW", "🟢", "risk-low"
    elif probability < 0.5:
        return "MEDIUM", "🟡", "risk-medium"
    elif probability < 0.75:
        return "HIGH", "🟠", "risk-high"
    else:
        return "CRITICAL", "🔴", "risk-critical"


def get_recommendation(risk_level):
    """Get maintenance recommendation based on risk level."""
    recommendations = {
        "LOW": "Continue normal operations. Monitor regularly.",
        "MEDIUM": "Schedule maintenance inspection within next cycle.",
        "HIGH": "Schedule immediate maintenance. Increase monitoring frequency.",
        "CRITICAL": "URGENT: Schedule emergency maintenance immediately. Consider equipment shutdown."
    }
    return recommendations.get(risk_level, "No recommendation available.")


def plot_feature_importance(model, feature_names, top_n=15):
    """Plot feature importance."""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[-top_n:]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(range(len(indices)), importances[indices], color='#1f77b4')
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels([feature_names[i] for i in indices])
        ax.set_xlabel('Importance')
        ax.set_title(f'Top {top_n} Feature Importances')
        plt.tight_layout()
        return fig
    return None


def plot_probability_gauge(probability):
    """Create a gauge chart for failure probability."""
    _, emoji, _ = get_risk_level(probability)
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"{emoji} Failure Probability", 'font': {'size': 24}},
        number={'suffix': "%", 'font': {'size': 40}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 30], 'color': "#d4edda"},
                {'range': [30, 50], 'color': "#fff3cd"},
                {'range': [50, 75], 'color': "#ffe5cc"},
                {'range': [75, 100], 'color': "#f8d7da"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': probability * 100
            }
        }
    ))
    
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
    return fig


def main():
    """Main application."""
    # Header
    st.markdown('<div class="main-header">🔧 Predictive Maintenance Engine</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI-Powered Equipment Failure Prediction using NASA Turbofan Dataset</div>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("📊 Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["🏠 Home", "🔮 Make Predictions", "📈 Model Performance", "ℹ️ About"]
    )
    
    # Load model
    model = load_model()
    
    if page == "🏠 Home":
        show_home_page(model)
    elif page == "🔮 Make Predictions":
        show_prediction_page(model)
    elif page == "📈 Model Performance":
        show_performance_page(model)
    elif page == "ℹ️ About":
        show_about_page()


def show_home_page(model):
    """Display home page."""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🎯 Purpose")
        st.write("""
        This application predicts equipment failures using machine learning, 
        enabling proactive maintenance scheduling and reducing downtime costs.
        """)
    
    with col2:
        st.markdown("### 🚀 Features")
        st.write("""
        - Binary failure classification
        - Imbalanced learning techniques
        - Time-series feature engineering
        - Interactive predictions
        """)
    
    with col3:
        st.markdown("### 📊 Dataset")
        st.write("""
        NASA Turbofan Jet Engine (C-MAPSS)
        - 100 engines (FD001)
        - 21 sensors + 3 operational settings
        - Run-to-failure simulation
        """)
    
    st.markdown("---")
    
    # Model Status
    st.markdown("### 🤖 Model Status")
    if model is not None:
        st.success("✅ Model loaded successfully!")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Model Type", model.__class__.__name__)
        with col2:
            st.metric("Features", f"{model.n_features_in_} (with rolling mean)")
        with col3:
            st.metric("Failure Threshold", f"{MODEL_CONFIG['failure_threshold']} cycles")
    else:
        st.error("❌ Model not found. Please train a model first.")
        st.code("python src/train.py --dataset FD001 --imbalance cost_sensitive")
    
    st.markdown("---")
    
    # Quick Start
    st.markdown("### 🚀 Quick Start")
    st.write("1. **Make Predictions**: Upload your data or use sample data")
    st.write("2. **View Performance**: Analyze model metrics and visualizations")
    st.write("3. **Understand Results**: Get risk levels and maintenance recommendations")


def show_prediction_page(model):
    """Display prediction page."""
    st.markdown("## 🔮 Make Predictions")
    
    if model is None:
        st.error("Model not loaded. Please train a model first.")
        return
    
    # Data source selection
    data_source = st.radio("Select Data Source", ["Sample Data", "Upload CSV"])
    
    if data_source == "Sample Data":
        dataset = st.selectbox("Select Dataset", ["FD001", "FD002", "FD003", "FD004"])
        
        if st.button("Load Sample Data"):
            with st.spinner("Loading data..."):
                train_df, test_df = load_sample_data(dataset)
                
                if test_df is not None:
                    st.session_state['data'] = test_df
                    st.success(f"Loaded {len(test_df)} samples from {dataset} test set")
    
    else:
        uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.session_state['data'] = df
                st.success(f"Loaded {len(df)} samples from uploaded file")
            except Exception as e:
                st.error(f"Error loading file: {e}")
    
    # Make predictions
    if 'data' in st.session_state:
        df = st.session_state['data']
        
        st.markdown("---")
        st.markdown("### Select Unit for Prediction")
        
        units = sorted(df['unit_number'].unique())
        selected_unit = st.selectbox("Unit Number", units)
        
        # Filter data for selected unit
        unit_data = df[df['unit_number'] == selected_unit].copy()
        
        # Show unit info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Cycles", len(unit_data))
        with col2:
            if 'RUL' in unit_data.columns:
                st.metric("Actual RUL", f"{unit_data.iloc[-1]['RUL']:.0f} cycles")
        with col3:
            if 'failure' in unit_data.columns:
                actual_failure = "Yes" if unit_data.iloc[-1]['failure'] == 1 else "No"
                st.metric("Actual Failure", actual_failure)
        
        # Predict button
        if st.button("🔮 Predict Failure", type="primary"):
            with st.spinner("Making prediction..."):
                # Get last row (most recent state)
                last_row = unit_data.iloc[[-1]]
                
                # Drop non-feature columns
                cols_to_drop = ['unit_number', 'time_in_cycles', 'RUL', 'failure']
                X = last_row.drop(columns=[col for col in cols_to_drop if col in last_row.columns])
                
                # Make prediction
                failure_prob = float(model.predict_proba(X)[0][1])
                failure_pred = bool(model.predict(X)[0])
                
                # Display results
                st.markdown("---")
                st.markdown("### 📊 Prediction Results")
                
                # Gauge chart
                fig = plot_probability_gauge(failure_prob)
                st.plotly_chart(fig, use_container_width=True)
                
                # Risk level and recommendation
                risk_level, emoji, css_class = get_risk_level(failure_prob)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"#### Risk Level: {emoji} <span class='{css_class}'>{risk_level}</span>", unsafe_allow_html=True)
                with col2:
                    st.markdown(f"#### Prediction: {'⚠️ Failure' if failure_pred else '✅ Normal'}")
                
                st.info(f"💡 **Recommendation:** {get_recommendation(risk_level)}")
                
                # Additional metrics
                if 'RUL' in unit_data.columns and 'failure' in unit_data.columns:
                    st.markdown("---")
                    st.markdown("### 🎯 Actual vs Predicted")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        actual = "Failure" if unit_data.iloc[-1]['failure'] == 1 else "Normal"
                        predicted = "Failure" if failure_pred else "Normal"
                        st.write(f"**Actual:** {actual}")
                        st.write(f"**Predicted:** {predicted}")
                        correct = (actual == predicted)
                        st.write(f"**Match:** {'✅ Correct' if correct else '❌ Incorrect'}")
                    
                    with col2:
                        st.write(f"**Actual RUL:** {unit_data.iloc[-1]['RUL']:.0f} cycles")
                        st.write(f"**Failure Probability:** {failure_prob:.2%}")


def show_performance_page(model):
    """Display model performance page."""
    st.markdown("## 📈 Model Performance")
    
    if model is None:
        st.error("Model not loaded. Please train a model first.")
        return
    
    # Load test data
    with st.spinner("Loading test data..."):
        train_df, test_df = load_sample_data("FD001")
    
    if test_df is None:
        st.error("Could not load test data.")
        return
    
    # Prepare data
    cols_to_drop = ['unit_number', 'time_in_cycles', 'RUL', 'failure']
    X_test = test_df.drop(columns=[col for col in cols_to_drop if col in test_df.columns])
    y_test = test_df['failure'] if 'failure' in test_df.columns else None
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    
    st.markdown("### 📊 Classification Metrics")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        acc = accuracy_score(y_test, y_pred)
        st.metric("Accuracy", f"{acc:.3f}")
    with col2:
        prec = precision_score(y_test, y_pred)
        st.metric("Precision", f"{prec:.3f}")
    with col3:
        rec = recall_score(y_test, y_pred)
        st.metric("Recall", f"{rec:.3f}")
    with col4:
        f1 = f1_score(y_test, y_pred)
        st.metric("F1-Score", f"{f1:.3f}")
    with col5:
        roc = roc_auc_score(y_test, y_pred_proba)
        st.metric("ROC-AUC", f"{roc:.3f}")
    
    st.markdown("---")
    
    # Confusion Matrix
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Confusion Matrix")
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test, y_pred)
        
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title('Confusion Matrix')
        st.pyplot(fig)
    
    with col2:
        st.markdown("### Feature Importance")
        fig = plot_feature_importance(model, X_test.columns, top_n=10)
        if fig:
            st.pyplot(fig)
        else:
            st.info("Feature importance not available for this model type.")


def show_about_page():
    """Display about page."""
    st.markdown("## ℹ️ About This Project")
    
    st.markdown("""
    ### 🎯 Project Overview
    This predictive maintenance system uses machine learning to forecast equipment failures, 
    enabling proactive maintenance scheduling and reducing downtime costs.
    
    ### 🏗️ Technical Stack
    - **Language**: Python 3.9+
    - **ML Framework**: Scikit-learn, XGBoost
    - **Data Processing**: Pandas, NumPy
    - **Visualization**: Matplotlib, Seaborn, Plotly
    - **Web Framework**: Streamlit
    - **API**: FastAPI
    
    ### 📊 Dataset
    **NASA Turbofan Jet Engine Dataset (C-MAPSS)**
    - Run-to-failure simulation data
    - 21 sensor measurements + 3 operational settings
    - Multiple operating conditions and fault modes
    
    ### 🔬 Methodology
    1. **Data Preprocessing**: MinMaxScaler normalization
    2. **Feature Engineering**: Rolling mean features (windows: 3, 5)
    3. **Imbalance Handling**: Cost-sensitive learning, SMOTE, undersampling
    4. **Models**: XGBoost, Random Forest
    5. **Evaluation**: Accuracy, Precision, Recall, F1, ROC-AUC
    
    ### 👤 Developer
    **Joyce Lapilus**
    - GitHub: [@atinyshrimp](https://github.com/atinyshrimp)
    - LinkedIn: [Joyce Lapilus](https://linkedin.com/in/joyce-lapilus)
    
    ### 📄 License
    MIT License - See LICENSE file for details
    """)


if __name__ == "__main__":
    main()

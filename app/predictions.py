"""Predictions page - Interactive failure prediction with timeline analysis."""

import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils import (
    load_model,
    load_removed_features,
    load_sample_data,
    load_rul_data,
    prepare_features_for_prediction,
    create_gauge_chart,
    create_prediction_timeline_chart,
)
from src.predict import make_predictions, get_risk_level
from src.config import MODEL_CONFIG

st.title("🔮 Failure Predictions")
st.markdown("Analyze equipment degradation patterns and predict failures")

# Load model
model = load_model()
removed_features = load_removed_features()

if model is None:
    st.error("❌ Failed to load model. Run the training pipeline first.")
    st.code("python src/train.py --dataset FD001 --imbalance cost_sensitive")
    st.stop()

# Load ground truth RUL data
rul_data = load_rul_data()

st.markdown("---")

# Input method selection
input_method = st.radio(
    "Select Input Method",
    ["📊 Sample Data (Test Set)", "✏️ Manual Input", "📁 Upload CSV"],
    horizontal=True
)

if input_method == "📊 Sample Data (Test Set)":
    st.subheader("🔬 Interactive Lifecycle Analysis")
    
    sample_df = load_sample_data()
    
    if sample_df is not None:
        # Select unit
        units = sorted(sample_df['unit_number'].unique()) if 'unit_number' in sample_df.columns else [1]
        selected_unit = st.selectbox("Select Equipment Unit", units)
        
        # Filter data for selected unit
        unit_data = sample_df[sample_df['unit_number'] == selected_unit].copy()
        total_cycles = len(unit_data)
        
        # Show ground truth if available
        if rul_data is not None:
            actual_rul = rul_data.iloc[selected_unit - 1]['RUL']  # 0-indexed
            actual_failure_cycle = total_cycles + actual_rul
            st.info(f"📊 **Unit {selected_unit}**: {total_cycles} cycles recorded | "
                   f"Actual RUL at final cycle: **{actual_rul} cycles** remaining")
        else:
            actual_rul = None
            st.write(f"**Unit {selected_unit}**: {total_cycles} time cycles available")
        
        st.markdown("---")
        
        # Analysis mode selection
        analysis_mode = st.radio(
            "Analysis Mode",
            ["🎯 Single Point Prediction", "📈 Timeline Analysis (Scrub Through Time)"],
            horizontal=True
        )
        
        if analysis_mode == "🎯 Single Point Prediction":
            st.markdown("### Select Prediction Window")
            
            col1, col2 = st.columns(2)
            with col1:
                # Let user choose where to predict
                min_cycle = 5  # Need at least 5 for rolling features
                prediction_cycle = st.slider(
                    "Predict at cycle",
                    min_value=min_cycle,
                    max_value=total_cycles,
                    value=total_cycles,
                    help="Select which point in the engine's lifecycle to make a prediction"
                )
            
            with col2:
                window_size = st.slider(
                    "Analysis window (cycles)",
                    min_value=5,
                    max_value=min(20, prediction_cycle),
                    value=min(10, prediction_cycle),
                    help="Number of previous cycles to use for feature calculation"
                )
            
            # Show lifecycle context
            lifecycle_pct = (prediction_cycle / total_cycles) * 100
            st.markdown(f"""
            **Lifecycle Context:** Analyzing cycle **{prediction_cycle}** of {total_cycles} 
            ({lifecycle_pct:.0f}% through recorded data)
            """)
            
            # Show data preview
            with st.expander("📋 View Selected Window Data"):
                window_data = unit_data.iloc[prediction_cycle - window_size:prediction_cycle]
                st.dataframe(window_data, width="stretch")
            
            # Prediction button
            if st.button("🚀 Predict Failure Risk", type="primary"):
                with st.spinner("Processing..."):
                    try:
                        # Get data up to selected cycle
                        window_data = unit_data.iloc[:prediction_cycle].tail(window_size).copy()
                        
                        # Prepare features
                        X = prepare_features_for_prediction(window_data, removed_features)
                        X_pred = X.iloc[[-1]]
                        
                        # Predict
                        probability, prediction, risk_level, recommendation = make_predictions(model, X_pred)
                        
                        # Display results
                        st.markdown("---")
                        st.subheader("🎯 Prediction Results")
                        
                        col1, col2 = st.columns([1, 1])
                        
                        with col1:
                            fig = create_gauge_chart(probability, f"Unit {selected_unit} @ Cycle {prediction_cycle}")
                            st.plotly_chart(fig, width="stretch")
                        
                        with col2:                            
                            st.metric("Failure Probability", f"{probability:.1%}")
                            st.metric("Risk Level", risk_level)
                            st.metric("Prediction Cycle", f"{prediction_cycle} / {total_cycles}")
                            
                            if risk_level == "LOW":
                                st.success(f"✅ {recommendation}")
                            elif risk_level == "MEDIUM":
                                st.warning(f"⚠️ {recommendation}")
                            elif risk_level == "HIGH":
                                st.error(f"🔶 {recommendation}")
                            else:
                                st.error(f"🚨 {recommendation}")
                        
                        # Ground truth comparison
                        if actual_rul is not None:
                            st.markdown("---")
                            st.subheader("📊 Ground Truth Comparison")
                            
                            cycles_remaining = total_cycles - prediction_cycle + actual_rul
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Model Prediction", 
                                         "⚠️ Failure Risk" if prediction else "✅ Normal")
                            with col2:
                                st.metric("Actual Cycles Remaining", f"{cycles_remaining}")
                            with col3:
                                # Check if model was correct (failure = <100 cycles remaining)
                                actually_failing = cycles_remaining < MODEL_CONFIG['failure_threshold']
                                model_correct = prediction == actually_failing
                                st.metric("Model Accuracy", "✅ Correct" if model_correct else "❌ Incorrect")
                            
                            if cycles_remaining < MODEL_CONFIG['failure_threshold']:
                                st.warning(f"🔴 This engine will fail within {cycles_remaining} cycles "
                                          f"(threshold: {MODEL_CONFIG['failure_threshold']} cycles)")
                            else:
                                st.success(f"🟢 This engine has {cycles_remaining} cycles remaining "
                                          f"before failure")
                        
                    except Exception as e:
                        st.error(f"Prediction error: {str(e)}")
                        st.exception(e)
        
        else:  # Timeline Analysis
            st.markdown("### 📈 Prediction Evolution Over Time")
            st.markdown("Watch how the model's failure prediction changes as the engine degrades")
            
            col1, col2 = st.columns(2)
            with col1:
                start_cycle = st.slider(
                    "Start from cycle",
                    min_value=5,
                    max_value=total_cycles - 5,
                    value=5
                )
            with col2:
                step_size = st.slider(
                    "Step size (cycles)",
                    min_value=1,
                    max_value=10,
                    value=5,
                    help="Analyze every N cycles"
                )
            
            if st.button("🚀 Generate Timeline", type="primary"):
                with st.spinner("Analyzing lifecycle..."):
                    try:
                        cycles = []
                        probabilities = []
                        
                        # Calculate predictions at each point
                        progress_bar = st.progress(0)
                        analysis_points = list(range(start_cycle, total_cycles + 1, step_size))
                        
                        for i, cycle in enumerate(analysis_points):
                            # Get data up to this cycle
                            window_data = unit_data.iloc[:cycle].tail(10).copy()
                            
                            # Prepare and predict
                            X = prepare_features_for_prediction(window_data, removed_features)
                            X_pred = X.iloc[[-1]]
                            prob = float(model.predict_proba(X_pred)[0][1])
                            
                            cycles.append(cycle)
                            probabilities.append(prob)
                            
                            progress_bar.progress((i + 1) / len(analysis_points))
                        
                        progress_bar.empty()
                        
                        # Create timeline chart
                        fig = create_prediction_timeline_chart(
                            cycles, probabilities,
                            f"Unit {selected_unit} - Failure Risk Evolution"
                        )
                        st.plotly_chart(fig, width="stretch")
                        
                        # Summary statistics
                        st.markdown("---")
                        st.subheader("📊 Timeline Summary")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Initial Risk", f"{probabilities[0]:.1%}")
                        with col2:
                            st.metric("Final Risk", f"{probabilities[-1]:.1%}")
                        with col3:
                            st.metric("Peak Risk", f"{max(probabilities):.1%}")
                        with col4:
                            # Find when it crossed 50% threshold
                            crossover = next((c for c, p in zip(cycles, probabilities, strict=True) if p > 0.5), None)
                            st.metric("Risk > 50% at", f"Cycle {crossover}" if crossover else "Never")
                        
                        # Ground truth overlay
                        if actual_rul is not None:
                            st.markdown("---")
                            st.info(f"**Ground Truth:** This engine's final recorded cycle was {total_cycles}. "
                                   f"It had **{actual_rul} cycles** remaining until failure. "
                                   f"Total life: {total_cycles + actual_rul} cycles.")
                        
                        # Detailed results table
                        with st.expander("📋 View Detailed Results"):
                            results_df = pd.DataFrame({
                                'Cycle': cycles,
                                'Failure Probability': [f"{p:.1%}" for p in probabilities],
                                'Risk Level': [get_risk_level(p)[0] for p in probabilities],
                                'Lifecycle %': [f"{(c/total_cycles)*100:.1f}%" for c in cycles]
                            })
                            st.dataframe(results_df, width="stretch")
                        
                    except Exception as e:
                        st.error(f"Analysis error: {str(e)}")
                        st.exception(e)
    else:
        st.warning("Sample data not found. Make sure test_FD001.txt exists in data/CMaps/")

elif input_method == "✏️ Manual Input":
    st.subheader("Manual Sensor Input")
    st.info("Enter 5 consecutive time steps of sensor data (each with 29 values)")
    
    st.markdown("""
    **Format**: 3 operational settings + 26 sensor measurements per time step
    
    Example values (comma-separated):
    ```
    -0.0007, -0.0004, 100.0, 518.67, 641.82, 1589.7, 1400.6, 14.62, 21.61, 554.36, 2388.02, 9046.19, 1.3, 47.47, 521.66, 2388.02, 8138.62, 8.4195, 0.03, 392, 2388, 100.0, 39.06, 23.4190, 0.0, 100.0, 0.0, 0.0, 0.0
    ```
    """)
    
    time_steps = []
    for i in range(5):
        with st.expander(f"Time Step {i+1}", expanded=(i == 0)):
            values = st.text_area(
                f"Enter 29 values for time step {i+1}",
                key=f"timestep_{i}",
                height=100,
                placeholder="Enter comma-separated values..."
            )
            if values:
                try:
                    parsed = [float(x.strip()) for x in values.split(',')]
                    if len(parsed) == 29:
                        time_steps.append(parsed)
                        st.success(f"✅ {len(parsed)} values parsed")
                    else:
                        st.warning(f"Expected 29 values, got {len(parsed)}")
                except ValueError:
                    st.error("Invalid format. Use comma-separated numbers.")
    
    if len(time_steps) == 5:
        if st.button("🚀 Predict Failure Risk", type="primary"):
            with st.spinner("Processing..."):
                try:
                    # Create DataFrame
                    columns = ["unit_number", "time_in_cycles"]
                    columns.extend([f"operational_setting_{i}" for i in range(1, 4)])
                    columns.extend([f"sensor_measurement_{i}" for i in range(1, 27)])
                    
                    rows = []
                    for idx, step in enumerate(time_steps):
                        row = [1, idx + 1] + step
                        rows.append(row)
                    
                    df = pd.DataFrame(rows, columns=columns)
                    
                    # Prepare and predict
                    X = prepare_features_for_prediction(df, removed_features)
                    X_pred = X.iloc[[-1]]
                    
                    probability, prediction, risk_level, recommendation = make_predictions(model, X_pred)
                    
                    # Display
                    st.markdown("---")
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        fig = create_gauge_chart(probability, "Failure Risk")
                        st.plotly_chart(fig, width="stretch")
                    
                    with col2:
                        st.metric("Failure Probability", f"{probability:.1%}")
                        st.metric("Risk Level", risk_level)
                        
                        if risk_level == "LOW":
                            st.success(f"✅ {recommendation}")
                        elif risk_level == "MEDIUM":
                            st.warning(f"⚠️ {recommendation}")
                        else:
                            st.error(f"🚨 {recommendation}")
                            
                except Exception as e:
                    st.error(f"Prediction error: {str(e)}")
    else:
        st.info(f"Enter all 5 time steps to enable prediction ({len(time_steps)}/5 complete)")

elif input_method == "📁 Upload CSV":
    st.subheader("Upload Sensor Data")
    
    st.markdown("""
    **CSV Format Requirements:**
    - Must have columns: `unit_number`, `time_in_cycles`, `operational_setting_1-3`, `sensor_measurement_1-26`
    - At least 5 rows (time steps) per unit
    """)
    
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Loaded {len(df)} rows")
            
            with st.expander("📋 Preview Data"):
                st.dataframe(df.head(10), width="stretch")
            
            # Select unit if multiple
            if 'unit_number' in df.columns:
                units = df['unit_number'].unique()
                selected_unit = st.selectbox("Select Unit to Predict", units)
                unit_data = df[df['unit_number'] == selected_unit].copy()
            else:
                unit_data = df.copy()
                unit_data['unit_number'] = 1
                unit_data['time_in_cycles'] = range(1, len(unit_data) + 1)
            
            # Add range selection for uploaded data too
            total_cycles = len(unit_data)
            
            col1, col2 = st.columns(2)
            with col1:
                prediction_cycle = st.slider(
                    "Predict at cycle",
                    min_value=5,
                    max_value=total_cycles,
                    value=total_cycles
                )
            with col2:
                window_size = st.slider(
                    "Analysis window",
                    min_value=5,
                    max_value=min(20, prediction_cycle),
                    value=min(10, prediction_cycle)
                )
            
            if st.button("🚀 Predict Failure Risk", type="primary"):
                with st.spinner("Processing..."):
                    if len(unit_data) < 5:
                        st.error("Need at least 5 time steps")
                    else:
                        window_data = unit_data.iloc[:prediction_cycle].tail(window_size).copy()
                        X = prepare_features_for_prediction(window_data, removed_features)
                        X_pred = X.iloc[[-1]]
                        
                        probability, prediction, risk_level, recommendation = make_predictions(model, X_pred)
                        
                        st.markdown("---")
                        col1, col2 = st.columns([1, 1])
                        
                        with col1:
                            fig = create_gauge_chart(probability, "Failure Risk")
                            st.plotly_chart(fig, width="stretch")
                        
                        with col2:
                            st.metric("Failure Probability", f"{probability:.1%}")
                            st.metric("Risk Level", risk_level)
                            st.metric("Analyzed Cycle", f"{prediction_cycle} / {total_cycles}")
                            st.info(recommendation)
                        
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")

# Footer
st.markdown("---")
st.caption("💡 **Tip**: Use Timeline Analysis to see how predictions evolve as the engine degrades - "
           "this demonstrates the model's ability to detect increasing failure risk over time.")

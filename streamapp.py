import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# ============================================
# PAGE CONFIGURATION
# ============================================

st.set_page_config(
    page_title="DeepCSAT: CSAT Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# CUSTOM CSS STYLING
# ============================================

st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    h1 {
        color: #1f77b4;
        font-size: 2.5em;
        margin-bottom: 10px;
    }
    h2 {
        color: #2c3e50;
        border-bottom: 3px solid #1f77b4;
        padding-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# LOAD MODEL AND SCALER
# ============================================

@st.cache_resource
def load_model_and_scaler():
    model = joblib.load('best_xgboost_csat_model.joblib')
    scaler = joblib.load('feature_scaler.joblib')
    return model, scaler

model, scaler = load_model_and_scaler()

# ============================================
# FEATURE NAMES & DESCRIPTIONS
# ============================================

FEATURE_NAMES = {
    'Response_Time': 'Response Time (minutes to first response)',
    'Resolution_Time': 'Resolution Time (hours to complete resolution)',
    'Agent_Professionalism': 'Agent Professionalism Score (1-10)',
    'Issue_Complexity': 'Issue Complexity Level (1-10)',
    'Communication_Quality': 'Communication Quality Score (1-10)',
    'Follow_up_Quality': 'Follow-up Quality Score (1-10)',
    'First_Contact_Resolution': 'First Contact Resolution Rate (%)',
    'Customer_Effort_Score': 'Customer Effort Score (1-10)'
}

FEATURE_GROUPS = {
    'Response Metrics': ['Response_Time', 'Resolution_Time'],
    'Service Quality': ['Agent_Professionalism', 'Communication_Quality'],
    'Issue Management': ['Issue_Complexity', 'Follow_up_Quality'],
    'Performance Indicators': ['First_Contact_Resolution', 'Customer_Effort_Score']
}

# ============================================
# SIDEBAR NAVIGATION
# ============================================

with st.sidebar:
    st.title("🎯 DeepCSAT")
    st.markdown("**E-Commerce Customer Satisfaction Prediction**")
    st.markdown("---")

    page = st.radio(
        "Navigation",
        ["🏠 Home", "🔮 Predictions", "📊 Analysis", "📈 Model Insights"]
    )

# ============================================
# PAGE 1: HOME
# ============================================

if page == "🏠 Home":
    st.title("🎯 DeepCSAT: E-Commerce Customer Satisfaction Score Prediction")
    st.markdown("### Powered by Deep Learning & XGBoost")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        ## Overview

        This advanced machine learning solution predicts **Customer Satisfaction (CSAT) scores** using deep learning
        and gradient boosting techniques. In the context of e-commerce, understanding customer satisfaction through
        their interactions and feedback is crucial for:

        - 📈 **Enhancing Service Quality** - Identify areas for improvement
        - 💰 **Improving Customer Retention** - Predict and retain at-risk customers
        - 🎯 **Business Growth** - Leverage insights for strategic decisions

        ## Key Features

        ✅ **Real-time Predictions** - Get CSAT scores instantly
        ✅ **Detailed Analysis** - Understand prediction drivers
        ✅ **Feature Importance** - Identify key satisfaction factors
        ✅ **Business Insights** - Actionable recommendations
        ✅ **Comprehensive Visualization** - Interactive charts and graphs
        """)

    with col2:
        st.info("""
        ### 📊 Model Statistics

        **Model Type:** XGBoost
        **Optimization:** GridSearchCV
        **Test MSE:** 2.02
        **Test R²:** 0.0059
        **Features:** 8
        **CSAT Scale:** 1-5
        **Status:** ✅ Active
        """)

    st.markdown("---")

    st.markdown("""
    ## Project Background

    Customer satisfaction in the e-commerce sector is a pivotal metric that influences loyalty, repeat business,
    and word-of-mouth marketing. Traditionally, companies have relied on direct surveys to gauge customer satisfaction,
    which can be time-consuming and may not always capture the full spectrum of customer experiences.

    With the advent of deep learning, it's now possible to predict customer satisfaction scores in real-time,
    offering a granular view of service performance and identifying areas for immediate improvement.
    """)

    st.markdown("---")

    # Objectives
    st.markdown("## 🎯 Specific Objectives")

    obj_cols = st.columns(3)

    with obj_cols[0]:
        st.markdown("""
        ### 1️⃣ Data Preparation
        Clean and preprocess the dataset to ensure it is
        suitable for training a deep learning model.
        """)

    with obj_cols[1]:
        st.markdown("""
        ### 2️⃣ Feature Engineering
        Identify and engineer features from the dataset
        that are most predictive of CSAT scores.
        """)

    with obj_cols[2]:
        st.markdown("""
        ### 3️⃣ Model Development
        Design and train a deep learning model to predict
        CSAT scores with high accuracy.
        """)

    obj_cols2 = st.columns(3)

    with obj_cols2[0]:
        st.markdown("""
        ### 4️⃣ Evaluation
        Assess the model's performance using appropriate
        metrics and validate its predictive accuracy.
        """)

    with obj_cols2[1]:
        st.markdown("""
        ### 5️⃣ Insight Generation
        Analyze the model's predictions to identify trends,
        patterns, and areas for service improvement.
        """)

    with obj_cols2[2]:
        st.markdown("""
        ### 6️⃣ Local Deployment
        Deploy the model in production to provide ongoing
        predictions and actionable insights.
        """)

# ============================================
# PAGE 2: PREDICTIONS
# ============================================

elif page == "🔮 Predictions":
    st.title("🔮 CSAT Score Prediction")
    st.markdown("Enter customer interaction features to predict satisfaction score")

    # Create tabs for different prediction modes
    tab1, tab2 = st.tabs(["Single Prediction", "Batch Prediction"])

    with tab1:
        st.markdown("### Single Customer Prediction")

        # Create columns for feature groups
        features = {}
        feature_list = []

        # Group 1: Response Metrics
        st.subheader("⏱️ Group 1: Response Metrics")
        col1, col2 = st.columns(2)

        with col1:
            response_time = st.slider(
                'Response Time (minutes)',
                min_value=0.0,
                max_value=60.0,
                value=15.0,
                step=0.5,
                help="Time in minutes for first response to customer query"
            )
            features['Response_Time'] = response_time
            feature_list.append(response_time)

        with col2:
            resolution_time = st.slider(
                'Resolution Time (hours)',
                min_value=0.0,
                max_value=48.0,
                value=8.0,
                step=0.5,
                help="Time in hours to completely resolve the issue"
            )
            features['Resolution_Time'] = resolution_time
            feature_list.append(resolution_time)

        st.divider()

        # Group 2: Service Quality
        st.subheader("⭐ Group 2: Service Quality")
        col3, col4 = st.columns(2)

        with col3:
            agent_prof = st.slider(
                'Agent Professionalism Score',
                min_value=1.0,
                max_value=10.0,
                value=7.5,
                step=0.5,
                help="Rate agent professionalism from 1-10"
            )
            features['Agent_Professionalism'] = agent_prof
            feature_list.append(agent_prof)

        with col4:
            comm_quality = st.slider(
                'Communication Quality Score',
                min_value=1.0,
                max_value=10.0,
                value=7.5,
                step=0.5,
                help="Rate communication quality from 1-10"
            )
            features['Communication_Quality'] = comm_quality
            feature_list.append(comm_quality)

        st.divider()

        # Group 3: Issue Management
        st.subheader("🛠️ Group 3: Issue Management")
        col5, col6 = st.columns(2)

        with col5:
            issue_complexity = st.slider(
                'Issue Complexity Level',
                min_value=1.0,
                max_value=10.0,
                value=5.0,
                step=0.5,
                help="Rate issue complexity from 1-10"
            )
            features['Issue_Complexity'] = issue_complexity
            feature_list.append(issue_complexity)

        with col6:
            followup_quality = st.slider(
                'Follow-up Quality Score',
                min_value=1.0,
                max_value=10.0,
                value=7.5,
                step=0.5,
                help="Rate follow-up quality from 1-10"
            )
            features['Follow_up_Quality'] = followup_quality
            feature_list.append(followup_quality)

        st.divider()

        # Group 4: Performance Indicators
        st.subheader("📊 Group 4: Performance Indicators")
        col7, col8 = st.columns(2)

        with col7:
            fcr_rate = st.slider(
                'First Contact Resolution Rate (%)',
                min_value=0.0,
                max_value=100.0,
                value=75.0,
                step=1.0,
                help="Percentage of issues resolved in first contact"
            )
            features['First_Contact_Resolution'] = fcr_rate
            feature_list.append(fcr_rate)

        with col8:
            ces_score = st.slider(
                'Customer Effort Score',
                min_value=1.0,
                max_value=10.0,
                value=6.0,
                step=0.5,
                help="How easy it was for customer to resolve issue (1-10)"
            )
            features['Customer_Effort_Score'] = ces_score
            feature_list.append(ces_score)

        st.markdown("---")

        col_predict, col_reset = st.columns(2)

        with col_predict:
            if st.button("🔮 Predict CSAT Score", use_container_width=True, key="predict_btn"):
                # Convert to numpy array
                features_array = np.array(feature_list).reshape(1, -1)

                # Scale features
                features_scaled = scaler.transform(features_array)

                # Make prediction
                prediction = model.predict(features_scaled)[0]

                # Store in session state
                st.session_state.prediction = prediction
                st.session_state.features = features

        with col_reset:
            if st.button("🔄 Reset Values", use_container_width=True):
                st.rerun()

        # Display results if prediction exists
        if 'prediction' in st.session_state:
            st.markdown("---")
            st.markdown("### 📊 Prediction Results")

            prediction = st.session_state.prediction

            # Create metrics display
            metric_col1, metric_col2, metric_col3 = st.columns(3)

            with metric_col1:
                st.metric(label="Predicted CSAT Score", value=f"{prediction:.2f}/5.00")

            with metric_col2:
                score_percentage = (prediction / 5.0) * 100
                st.metric(label="Satisfaction Level", value=f"{score_percentage:.1f}%")

            with metric_col3:
                if prediction >= 4:
                    sentiment = "Very Satisfied"
                    emoji = "😊"
                elif prediction >= 3:
                    sentiment = "Satisfied"
                    emoji = "😐"
                elif prediction >= 2:
                    sentiment = "Neutral"
                    emoji = "😕"
                else:
                    sentiment = "Dissatisfied"
                    emoji = "😞"
                st.metric(label="Customer Sentiment", value=f"{emoji} {sentiment}")

            # Satisfaction gauge
            st.markdown("### Satisfaction Gauge")

            fig, ax = plt.subplots(figsize=(10, 2))
            colors_gauge = ['#d62728' if prediction < 2 else '#ff7f0e' if prediction < 3 else '#2ca02c' if prediction < 4 else '#1f77b4']
            ax.barh(['CSAT Score'], [prediction], color=colors_gauge, height=0.3)
            ax.set_xlim(0, 5)
            ax.set_xlabel('Score (1-5)', fontsize=12)
            ax.text(prediction/2, 0, f'{prediction:.2f}', ha='center', va='center', fontsize=14, fontweight='bold', color='white')
            plt.tight_layout()
            st.pyplot(fig)

            # Input features visualization
            st.markdown("### Input Features Summary")
            features_df = pd.DataFrame({
                'Feature': list(st.session_state.features.keys()),
                'Value': list(st.session_state.features.values()),
                'Description': [FEATURE_NAMES[f] for f in st.session_state.features.keys()]
            })

            st.dataframe(features_df, use_container_width=True)

            col_chart1, col_chart2 = st.columns(2)

            with col_chart1:
                fig, ax = plt.subplots(figsize=(8, 4))
                features_short = [k.replace('_', '\n') for k in st.session_state.features.keys()]
                ax.bar(range(len(features_short)), list(st.session_state.features.values()), color='#2ecc71')
                ax.set_ylabel('Feature Value')
                ax.set_title('Feature Values Visualization')
                ax.set_xticks(range(len(features_short)))
                ax.set_xticklabels(features_short, fontsize=8)
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                st.pyplot(fig)

            with col_chart2:
                fig, ax = plt.subplots(figsize=(8, 4))
                for i, (k, v) in enumerate(st.session_state.features.items()):
                    ax.barh(i, v, color='#3498db', alpha=0.7)
                ax.set_yticks(range(len(st.session_state.features)))
                ax.set_yticklabels([k.replace('_', ' ') for k in st.session_state.features.keys()], fontsize=9)
                ax.set_xlabel('Value')
                ax.set_title('Feature Values (Horizontal)')
                plt.tight_layout()
                st.pyplot(fig)

    with tab2:
        st.markdown("### Batch Prediction (Multiple Customers)")

        st.info("Upload a CSV file with 8 columns matching the feature names for batch prediction")

        # Download sample CSV
        sample_data = {
            'Response_Time': [10, 15, 20],
            'Resolution_Time': [4, 8, 12],
            'Agent_Professionalism': [8, 7, 6],
            'Issue_Complexity': [3, 5, 7],
            'Communication_Quality': [8, 7, 6],
            'Follow_up_Quality': [7, 6, 5],
            'First_Contact_Resolution': [80, 70, 60],
            'Customer_Effort_Score': [7, 6, 5]
        }
        sample_df = pd.DataFrame(sample_data)

        csv = sample_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Sample CSV",
            data=csv,
            file_name="sample_csat_data.csv",
            mime="text/csv"
        )

        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)

            if df.shape[1] >= 8:
                features_batch = df.iloc[:, :8].values
                features_scaled_batch = scaler.transform(features_batch)
                predictions_batch = model.predict(features_scaled_batch)

                results_df = df.copy()
                results_df['Predicted_CSAT'] = predictions_batch
                results_df['Sentiment'] = results_df['Predicted_CSAT'].apply(
                    lambda x: '😊 Very Satisfied' if x >= 4 else '😐 Satisfied' if x >= 3 else '😕 Neutral' if x >= 2 else '😞 Dissatisfied'
                )

                st.dataframe(results_df, use_container_width=True)

                # Download results
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Predictions",
                    data=csv,
                    file_name="csat_predictions.csv",
                    mime="text/csv"
                )

                # Statistics
                st.markdown("### Batch Prediction Statistics")
                stat_col1, stat_col2, stat_col3 = st.columns(3)

                with stat_col1:
                    st.metric("Average CSAT", f"{predictions_batch.mean():.2f}")
                with stat_col2:
                    st.metric("Min CSAT", f"{predictions_batch.min():.2f}")
                with stat_col3:
                    st.metric("Max CSAT", f"{predictions_batch.max():.2f}")
            else:
                st.error("CSV must have at least 8 columns matching the features")

# ============================================
# PAGE 3: ANALYSIS
# ============================================

elif page == "📊 Analysis":
    st.title("📊 Detailed Analysis & Insights")

    st.markdown("### Feature Importance Analysis")

    importance_data = {
        'Feature': ['First_Contact_Resolution', 'Customer_Effort_Score', 'Issue_Complexity', 'Follow_up_Quality',
                    'Communication_Quality', 'Agent_Professionalism', 'Resolution_Time', 'Response_Time'],
        'Importance': [0.151819, 0.146843, 0.135437, 0.132161, 0.128253, 0.118521, 0.096417, 0.090551],
        'Description': [
            'First Contact Resolution Rate (%)',
            'Customer Effort Score (1-10)',
            'Issue Complexity Level (1-10)',
            'Follow-up Quality Score (1-10)',
            'Communication Quality Score (1-10)',
            'Agent Professionalism Score (1-10)',
            'Resolution Time (hours)',
            'Response Time (minutes)'
        ]
    }

    importance_df = pd.DataFrame(importance_data)

    col_imp, col_info = st.columns([2, 1])

    with col_imp:
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = plt.cm.viridis(np.linspace(0, 1, len(importance_df)))
        bars = ax.barh(importance_df['Feature'], importance_df['Importance'], color=colors)
        ax.set_xlabel('Importance Score', fontsize=12)
        ax.set_title('Feature Importance for CSAT Prediction', fontsize=14, fontweight='bold')

        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, importance_df['Importance'])):
            ax.text(val + 0.003, bar.get_y() + bar.get_height()/2, f'{val:.4f}',
                   ha='left', va='center', fontsize=9, fontweight='bold')

        plt.tight_layout()
        st.pyplot(fig)

    with col_info:
        st.info("""
        ### 🎯 Top Contributing Features

        **#1: First Contact Resolution** (15.2%)
        - Most critical CSAT driver
        - Resolving on first contact highly satisfies customers

        **#2: Customer Effort** (14.7%)
        - Second most important
        - Ease of resolution matters

        **#3: Issue Complexity** (13.5%)
        - How complex the problem affects satisfaction
        """)

    st.markdown("---")
    st.markdown("### Model Performance Metrics")

    metrics_data = {
        'Metric': ['MSE', 'MAE', 'R² Score'],
        'Training': [0.1739, 0.3337, 0.9123],
        'Testing': [2.2601, 1.2954, -0.1137],
        'Optimized (Best)': [2.0174, 1.2283, 0.0059]
    }

    metrics_df = pd.DataFrame(metrics_data)
    st.dataframe(metrics_df, use_container_width=True)

    # Model comparison
    st.markdown("### Model Comparison (ANN vs LSTM vs XGBoost)")

    col_m1, col_m2, col_m3 = st.columns(3)

    with col_m1:
        st.markdown("""
        ### ANN (Model 1)

        **Test MSE:** 2.3682
        **Test R²:** -0.1670
        **Status:** ❌ Weak

        - Deep learning approach
        - Poor generalization
        - Overfitting issues
        """)

    with col_m2:
        st.markdown("""
        ### LSTM (Model 2)

        **Test MSE:** 2.6152
        **Test R²:** -0.2513
        **Status:** ❌ Weak

        - Sequential learning
        - Best for time-series
        - Not suitable here
        """)

    with col_m3:
        st.markdown("""
        ### XGBoost (Model 3) ✅

        **Test MSE:** 2.0174
        **Test R²:** 0.0059
        **Status:** ✅ Selected

        - Gradient boosting
        - Best performance
        - Interpretable
        """)

# ============================================
# PAGE 4: MODEL INSIGHTS
# ============================================

elif page == "📈 Model Insights":
    st.title("📈 Model Insights & Technical Details")

    st.markdown("### Model Architecture")

    col_arch, col_details = st.columns([1, 1])

    with col_arch:
        st.markdown("""
        #### XGBoost Configuration

        **Algorithm:** Gradient Boosting Regressor
        **Framework:** XGBoost
        **Optimization:** GridSearchCV
        **Cross-Validation:** 3-Fold

        #### Hyperparameters

        - n_estimators: 100
        - max_depth: 6
        - learning_rate: 0.1
        - subsample: 0.8
        - colsample_bytree: 0.8
        """)

    with col_details:
        st.markdown("""
        #### Model Performance

        **Training MSE:** 0.1739
        **Training MAE:** 0.3337
        **Training R²:** 0.9123

        **Testing MSE:** 2.2601
        **Testing MAE:** 1.2954
        **Testing R²:** -0.1137

        **Optimized MSE:** 2.0174
        **Optimized R²:** 0.0059
        """)

    st.markdown("---")

    st.markdown("### Evaluation Metrics Explanation")

    tab1, tab2, tab3 = st.tabs(["MSE", "MAE", "R² Score"])

    with tab1:
        st.markdown("""
        #### Mean Squared Error (MSE)

        **Definition:** Average of squared prediction errors

        **Formula:** MSE = (1/n) Σ(y_actual - y_predicted)²

        **Business Impact:**
        - Penalizes large prediction errors heavily
        - Lower values indicate better accuracy
        - Our MSE of 2.02 means predictions deviate ~1.42 points on average

        **What it means for CSAT:**
        - Customers rated 3 might be predicted as 1 or 5
        - High MSE indicates potential misallocation of resources
        """)

    with tab2:
        st.markdown("""
        #### Mean Absolute Error (MAE)

        **Definition:** Average absolute difference between predictions and actual values

        **Formula:** MAE = (1/n) Σ|y_actual - y_predicted|

        **Business Impact:**
        - More interpretable than MSE
        - Direct representation of average error
        - Our MAE of 1.23 represents ~25% error on 5-point scale

        **What it means for CSAT:**
        - Predictions are off by ~1.2 points on average
        - Useful for identifying moderately satisfied vs dissatisfied customers
        """)

    with tab3:
        st.markdown("""
        #### R² Score (Coefficient of Determination)

        **Definition:** Proportion of variance in CSAT explained by the model

        **Range:** -∞ to 1 (1 is perfect, 0 is mean baseline)

        **Our Score:** 0.0059 (Positive but weak)

        **Business Impact:**
        - R² > 0: Model better than mean prediction
        - Our model explains only 0.59% of variance
        - Suggests need for better features or data

        **Interpretation:**
        - Model is marginally useful
        - Should focus on feature engineering for improvement
        """)

    st.markdown("---")

    st.markdown("### Feature Descriptions")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **Response Time**
        - Time taken to first respond to customer
        - Measured in minutes
        - Lower is better for satisfaction

        **Resolution Time**
        - Time taken to fully resolve issue
        - Measured in hours
        - Impacts customer satisfaction

        **Agent Professionalism**
        - Quality of agent behavior
        - Rated 1-10 scale
        - Higher improves CSAT
        """)

    with col2:
        st.markdown("""
        **Issue Complexity**
        - Difficulty level of problem
        - Rated 1-10 scale
        - More complex = harder to satisfy

        **Communication Quality**
        - Clarity and helpfulness of agent
        - Rated 1-10 scale
        - Critical for satisfaction

        **First Contact Resolution**
        - % of issues resolved first contact
        - 0-100% scale
        - Top satisfaction driver
        """)

st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 30px; background: linear-gradient(135deg, #1f77b4 0%, #2c3e50 100%); border-radius: 10px; color: white; margin-top: 20px;'>
    <h3 style='color: white; margin: 0;'>🚀 DeepCSAT v1.0</h3>
    <p style='color: #ecf0f1; margin: 10px 0;'><strong>E-Commerce Customer Satisfaction Prediction System</strong></p>
    <p style='color: #ecf0f1; margin: 10px 0;'>Powered by XGBoost & Streamlit | © 2025</p>
</div>
""", unsafe_allow_html=True)

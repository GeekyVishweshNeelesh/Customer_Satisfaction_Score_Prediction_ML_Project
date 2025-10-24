# 🎯 DeepCSAT: E-Commerce Customer Satisfaction Score Prediction

An advanced machine learning solution that predicts Customer Satisfaction (CSAT) scores for e-commerce businesses using deep learning and gradient boosting techniques. This project implements and compares three ML models (ANN, LSTM, XGBoost) and deploys the best performing model through an interactive Streamlit web application.

## 📊 Overview

**DeepCSAT** is an intelligent system that predicts customer satisfaction scores based on 8 key customer interaction features. The project follows a comprehensive ML pipeline:

1. **Data Preparation** - Clean and preprocess customer data
2. **Feature Engineering** - Identify predictive features for CSAT
3. **Model Development** - Train 3 different ML models
4. **Hyperparameter Optimization** - Tune models using GridSearchCV & RandomizedSearchCV
5. **Model Evaluation** - Compare performance using MSE, MAE, R²
6. **Deployment** - Deploy best model via Streamlit application.

### Key Metrics

| Metric | Value | Status |
|--------|-------|--------|
| **Best Model** | XGBoost | ✅ Selected |
| **Test MSE** | 2.0174 | Optimized |
| **Test R² Score** | 0.0059 | Positive |
| **MSE Improvement** | 10.67% | vs Baseline |
| **Features** | 8 | Comprehensive |
| **CSAT Scale** | 1-5 | Standard |


## ✨ Features

- ✅ **Real-time CSAT Predictions** - Get satisfaction scores instantly
- ✅ **Single Customer Prediction** - Interactive feature input with sliders
- ✅ **Batch Predictions** - Upload CSV for multiple customers
- ✅ **Feature Importance Analysis** - Understand satisfaction drivers
- ✅ **Model Insights** - Technical details and metric explanations
- ✅ **Interactive Visualizations** - Charts, gauges, and comparisons
- ✅ **Download Predictions** - Export results as CSV
- ✅ **Comprehensive Documentation** - Built-in guides and descriptions

## 🎓 Project Background

Customer satisfaction in e-commerce is crucial for:
- **Customer Loyalty** - Satisfied customers become repeat buyers
- **Business Growth** - Positive word-of-mouth marketing
- **Competitive Advantage** - Better retention than competitors

Traditional survey-based approaches are time-consuming and reactive. DeepCSAT uses machine learning to:
- Predict CSAT scores in real-time
- Provide granular service performance insights
- Enable proactive service improvements
- Support data-driven decision making


## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- 200 MB free disk space.




### File Descriptions

#### `streamapp.py` (Main Application)
- Streamlit web application
- 4-page interactive interface
- Real-time predictions
- Data visualization
- ~450 lines of code

#### `best_xgboost_csat_model.joblib`
- Production-ready XGBoost model
- GridSearchCV optimized
- Trained on 1000 samples
- File size: ~50 KB

#### `feature_scaler.joblib`
- StandardScaler for feature normalization
- Fitted on training data
- Essential for preprocessing new predictions
- File size: ~2 KB

#### `Proj11__DeepCSAT.ipynb`
- Complete ML pipeline in Jupyter notebook
- Data preparation and exploration
- All 3 models implementation (ANN, LSTM, XGBoost)
- Hyperparameter tuning (GridSearchCV, RandomizedSearchCV)
- Model evaluation and comparison
- Results visualization



## 🤖 Models Implemented

### Model 1: Artificial Neural Network (ANN)
```
Architecture: 8 → 64 → 32 → 16 → 1
Activation: ReLU (hidden), Linear (output)
Framework: TensorFlow/Keras
```

**Performance:**
- Test MSE: 2.3682
- Test R²: -0.1670 ❌
- **Issue**: Severe overfitting (Train R² = 0.9123)
- **Status**: Not selected

### Model 2: Long Short-Term Memory (LSTM)
```
Architecture: 10 timesteps → LSTM(64) → LSTM(32) → Dense(16) → 1
Framework: TensorFlow/Keras
```

**Performance:**
- Test MSE: 2.6152
- Test R²: -0.2513 ❌
- **Issue**: Designed for sequential data, not suitable for this structured dataset
- **Status**: Not selected

### Model 3: XGBoost (SELECTED) ✅
```
Algorithm: Gradient Boosting Regressor
Trees: 100 estimators
Optimization: GridSearchCV (3-fold CV)
```


**Hyperparameters:**
- max_depth: 6
- learning_rate: 0.1
- subsample: 0.8
- colsample_bytree: 0.8


**Performance:**
- Test MSE: 2.0174 ✅
- Test R²: 0.0059 ✅
- **Advantage**: Best generalization, interpretable, production-ready
- **Status**: Selected for deployment.



## 📈 Results & Performance

### Model Comparison

| Metric | ANN | LSTM | XGBoost |
|--------|-----|------|---------|
| Test MSE | 2.3682 | 2.6152 | 2.0174 ✅ |
| Test R² | -0.1670 | -0.2513 | 0.0059 ✅ |
| Test MAE | 1.2960 | 1.3720 | 1.2954 |
| Status | ❌ Poor | ❌ Poor | ✅ Selected |



## 💻 Streamlit App

### Application Pages

#### 🏠 Home
- Project overview and objectives
- Business background and motivation
- Model statistics dashboard
- Key features summary

#### 🔮 Predictions
**Single Prediction Mode:**
- 8 interactive sliders organized in 4 groups
- Real-time CSAT score prediction
- Satisfaction sentiment analysis
- Visual gauge display
- Feature summary table
- Input visualization charts

**Batch Prediction Mode:**
- CSV file upload
- Sample data download
- Bulk predictions for multiple customers
- Results export as CSV
- Batch statistics

#### 📊 Analysis
- Feature importance visualization
- Model performance comparison (ANN vs LSTM vs XGBoost)
- Evaluation metrics analysis
- Model selection rationale

#### 📈 Model Insights
- XGBoost architecture details
- Hyperparameter configuration
- Evaluation metrics explanation (MSE, MAE, R²)
- Feature descriptions and ranges


## 💡 Key Findings

1. **Response Quality is Critical**
   - Features 6 & 7 account for ~30% of satisfaction
   - Focus on response speed and communication

2. **First Contact Matters**
   - Highest importance feature (15.2%)
   - Resolve issues on first contact when possible

3. **Effort Minimization Important**
   - Customer Effort Score is 2nd most important (14.7%)
   - Make resolution process easy for customers

4. **Model Selection**
   - XGBoost outperformed deep learning models
   - Ensemble methods work better for structured business data
   - 10.67% MSE improvement through optimization.
  

## 📚 Technical Stack

| Component | Technology | Version |
|-----------|-----------|---------|
| Language | Python | 3.8+ |
| Web Framework | Streamlit | 1.28+ |
| ML Libraries | XGBoost, scikit-learn, TensorFlow | Latest |
| Data Processing | Pandas, NumPy | Latest |
| Visualization | Matplotlib, Seaborn | Latest |
| Serialization | Joblib | 1.3+ |

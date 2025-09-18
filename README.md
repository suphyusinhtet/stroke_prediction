# 🧠 Brain Stroke Prediction using Machine Learning

A comprehensive analysis and machine learning approach for predicting stroke likelihood using demographic and health-related data.

## 📊 Project Overview

This project conducts an in-depth analysis of the [Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset) to identify key predictors of stroke and develop models that can support early detection and prevention efforts.

### Key Objectives
- Understand the structure and distribution of stroke-related features
- Handle missing values and data imbalances effectively  
- Identify patterns and correlations in health indicators
- Build and compare multiple machine learning models
- Achieve high recall for stroke detection (prioritizing identification of actual stroke cases)

## 🔍 Dataset Analysis

### Data Characteristics
- **Total Records**: 5,110 patient records
- **Features**: 11 attributes including demographic and health indicators
- **Target Variable**: Stroke occurrence (binary: 0/1)
- **Class Imbalance**: Only 4.87% positive stroke cases

### Key Findings from EDA

#### Missing Data Patterns
- **BMI**: Contains actual missing (NaN) values
- **Smoking Status**: 30.22% marked as "Unknown"
- Missing BMI data shows strong correlation with stroke occurrence
- Missing data is **not** missing at random - systematic patterns identified

#### Feature Distributions
- **Age**: Strong positive correlation with stroke (primary risk factor)
- **Glucose Level**: Moderate influence on stroke likelihood  
- **BMI**: Less predictive power than expected
- **Gender, Hypertension, Heart Disease**: Significant correlations with stroke

## 🛠️ Data Preprocessing

### Data Cleaning Steps
1. **Removed** `id` column (non-informative)
2. **Removed** single "Other" gender entry (insufficient data)
3. **BMI Categorization**: Converted continuous BMI to categories:
   - Underweight (≤18.5)
   - Normal weight (18.5-25)
   - Overweight (25-30)
   - Obese (>30)
   - Child (<20 years)
   - Unknown (missing values)

4. **Smoking Status**: Corrected "Unknown" to "never smoked" for children ≤12 years
5. **Feature Engineering**:
   - Binary encoding for categorical variables
   - Normalization of continuous variables
   - One-hot encoding for multi-categorical features

### Handling Class Imbalance
- **Applied SMOTE** (Synthetic Minority Oversampling Technique)
- Balanced training set from 4.87% to 50% positive cases
- **Important**: Oversampling applied only to training data, test set remains unchanged

## 🤖 Machine Learning Models

### Models Evaluated
1. **Logistic Regression** (baseline)
2. **Decision Tree Classifier**  
3. **XGBoost Classifier**
4. **Random Forest Classifier**
5. **Tuned Logistic Regression** (hyperparameter optimized)

### Model Optimization
- **Grid Search with Cross-Validation** for hyperparameter tuning
- **Pipeline Approach** to properly handle SMOTE during cross-validation
- **F1-weighted scoring** for model selection

#### Best Hyperparameters Found:
- **Decision Tree**: `max_depth=3, criterion='entropy'`
- **XGBoost**: `learning_rate=0.001, max_depth=2, n_estimators=200`
- **Random Forest**: `n_estimators=500, max_depth=2, bootstrap=True`
- **Tuned Logistic Regression**: `C=0.01, penalty='l2', solver='liblinear'`

## 📈 Results

### Model Performance Comparison

| Model | Accuracy | AUC | Precision | Recall | F1-Score |
|-------|----------|-----|-----------|--------|----------|
| **Logistic Regression (No SMOTE)** | 94% | 0.85 | 0.5 | 0.02 | 0.91 |
| **Logistic Regression (After SMOTE)** | 82% | 0.80 | 0.16 | 0.45 | 0.86 |
| **Decision Tree** | 85% | 0.65 | 0.17 | 0.35 | 0.88 |
| **XGBoost** | 66.14% | 0.83 | 0.13 | 0.89 | 0.75 |
| **Random Forest** | 66.14% | 0.84 | 0.14 | 0.89 | 0.75 |
| **Logistic Regression (Tuned)** | 78% | 0.83 | 0.18 | 0.71 | 0.83 |

### Key Performance Insights

#### 🎯 Best Models for Stroke Detection:
1. **XGBoost**: Highest recall (92%) - best at identifying actual stroke cases
2. **Random Forest**: Strong recall (89%) with good AUC (0.84)
3. **Tuned Logistic Regression**: Balanced performance across all metrics

#### 📊 Trade-offs Analysis:
- **High Recall Models** (XGBoost, Random Forest): Excel at detecting true positives but produce more false positives
- **Balanced Models** (Tuned Logistic Regression): Better precision-recall balance
- **Medical Context**: High recall preferred due to serious consequences of missing stroke cases

## 🔬 Clinical Relevance

### Why High Recall Matters
In medical diagnosis, **false negatives are more critical than false positives**:
- Missing a stroke case (false negative) can be life-threatening
- False positives lead to additional testing but ensure patient safety
- Models prioritizing recall help ensure no stroke cases are missed

### Feature Importance
Based on the analysis, key stroke predictors include:
1. **Age** (strongest predictor)
2. **Hypertension** 
3. **Heart Disease**
4. **Average Glucose Level**
5. **BMI Category** (including "Unknown" status)
6. **Smoking Status**

## 📁 Repository Structure

```
stroke-prediction/
│
├── data/
│   └── healthcare-dataset-stroke-data.csv
│
├── notebooks/
│   └── stroke_analysis.ipynb
│
├── visualizations/
│   ├── numerical_distributions.png
│   ├── categorical_distributions.png
│   ├── correlation_matrix_after.png
│   ├── roc_curves_comparison.png
│   ├── pr_curves_comparison.png
│   └── metrics_comparison.png
│
├── models/
│   └── trained_models.pkl
│
└── README.md
```

## 🚀 Getting Started

### Prerequisites
```python
pandas>=1.3.0
numpy>=1.20.0
scikit-learn>=1.0.0
xgboost>=1.5.0
plotly>=5.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
imbalanced-learn>=0.8.0
```

### Installation
```bash
git clone https://github.com/suphyusinhtet/stroke-prediction.git
cd stroke-prediction
pip install -r requirements.txt
```

### Usage
```python
# Load the trained model
import pickle
with open('models/best_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Make predictions
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)
```

## 📊 Visualizations

The project includes comprehensive visualizations:
- Feature distribution analysis
- Correlation matrices  
- ROC and Precision-Recall curves
- Model performance comparisons
- Radar charts for metric comparison

## 🔮 Future Improvements

1. **Feature Engineering**: Create additional risk score features
2. **Advanced Models**: Experiment with neural networks and ensemble methods
3. **Cross-Validation**: Implement more robust validation strategies
4. **Clinical Validation**: Collaborate with medical professionals for validation
5. **Real-time Prediction**: Develop API for real-time stroke risk assessment

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Dataset provided by [fedesoriano](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset) on Kaggle
- Medical guidelines from CDC for BMI categorization
- Research insights from stroke prevention studies

## 📞 Contact

For questions or collaborations, please reach out:
- **Email**: your.email@example.com
- **LinkedIn**: [Your LinkedIn Profile](https://linkedin.com/in/su-phyu-sin-htet)
- **GitHub**: [@suphyusinhtet](https://github.com/suphyusinhtet)

---

**⚠️ Medical Disclaimer**: This project is for educational and research purposes only. The models and predictions should not be used for actual medical diagnosis or treatment decisions. Always consult qualified healthcare professionals for medical advice.

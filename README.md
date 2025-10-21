# Space Decay Prediction - Machine Learning Project

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

A comprehensive machine learning project for analyzing satellite and space debris orbital data to predict object characteristics and understand decay patterns.

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Project Pipeline](#project-pipeline)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Results](#results)
- [Model Performance](#model-performance)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Key Learnings](#key-learnings)

## 🎯 Project Overview

This project implements a complete data science pipeline to analyze space debris and satellite orbital data. The main objective is to classify different types of space objects (DEBRIS, PAYLOAD, ROCKET BODY, TBA) based on their orbital parameters and physical characteristics.

**Key Features:**
- ✅ Complete data preprocessing and cleaning pipeline
- ✅ Comprehensive exploratory data analysis (EDA)
- ✅ Principal Component Analysis (PCA) for dimensionality reduction
- ✅ Multiple machine learning models comparison
- ✅ Advanced model evaluation with cross-validation
- ✅ Production-ready model deployment

## 📊 Dataset

**Source:** `space_decay.csv`

**Size:** 14,372 rows × 40 columns

**Key Features:**
- **Orbital Parameters:** Mean motion, eccentricity, inclination, semi-major axis, period
- **Physical Characteristics:** BSTAR drag term, RCS size
- **Metadata:** Object name, country code, launch date, classification type

**Target Variable:** `OBJECT_TYPE` (4 classes)
- DEBRIS: 8,431 samples (58.7%)
- PAYLOAD: 4,950 samples (34.4%)
- ROCKET BODY: 744 samples (5.2%)
- TBA: 247 samples (1.7%)

## 🔄 Project Pipeline

### 1. Data Loading & Exploration

- Loaded 14,372 orbital records with 40 features
- Performed initial data profiling and statistical analysis
- Identified data types and distributions

### 2. Data Cleaning

**Missing Values Handling:**
- Dropped `DECAY_DATE` column (100% missing)
- Filled numerical missing values with median (robust to outliers)
- Filled categorical missing values with mode

**Key Statistics:**
- Original missing values: 6 columns with missing data
- Final missing values: 0 (100% complete dataset)
- Duplicate rows: 0

**Outlier Detection:**
- Applied IQR (Interquartile Range) method
- Identified outliers in:
  - `MEAN_MOTION`: 2,371 outliers
  - `ECCENTRICITY`: 2,025 outliers
  - `NORAD_CAT_ID`: 39 outliers
- **Decision:** Retained outliers as they represent legitimate space phenomena (highly eccentric orbits)

### 3. Data Preprocessing

**Feature Selection:**
Selected 15 relevant features for modeling:
- Numerical: `MEAN_MOTION`, `ECCENTRICITY`, `INCLINATION`, `RA_OF_ASC_NODE`, `ARG_OF_PERICENTER`, `MEAN_ANOMALY`, `BSTAR`, `MEAN_MOTION_DOT`, `SEMIMAJOR_AXIS`, `PERIOD`, `APOAPSIS`, `PERIAPSIS`, `REV_AT_EPOCH`
- Categorical: `COUNTRY_CODE`, `CLASSIFICATION_TYPE`

**Encoding:**
- Label encoding for categorical variables
- Target encoding for multi-class classification

**Scaling:**
- StandardScaler applied to all numerical features
- Fitted on training data only to prevent data leakage

### 4. Exploratory Data Analysis (EDA)

**Correlation Analysis:**
- Identified highly correlated feature pairs (|r| > 0.8):
  - `MEAN_MOTION` ↔ `SEMIMAJOR_AXIS`: -0.871
  - `MEAN_MOTION` ↔ `APOAPSIS`: -0.879
  - `SEMIMAJOR_AXIS` ↔ `PERIOD`: 0.926
  - `SEMIMAJOR_AXIS` ↔ `APOAPSIS`: 0.949
  - `PERIOD` ↔ `APOAPSIS`: 0.883

**Visualizations:**
- Distribution plots for numerical features
- Box plots for outlier detection
- Correlation heatmap
- Scatter plots for feature relationships
- Target variable distribution analysis

### 5. Principal Component Analysis (PCA)

**Dimensionality Reduction:**
- Original features: 13 numerical features
- Components for 95% variance: 10 components
- Components for 90% variance: 8 components
- **Reduction ratio:** ~23% reduction while preserving 95% variance

**Key Insights:**
- First 2 PCs explain ~60% of total variance
- First 3 PCs explain ~72% of total variance
- Visualized high-dimensional data in 2D/3D space
- Identified feature contributions to principal components

### 6. Feature Engineering

**Created New Features:**
- **Orbit Eccentricity Category:** Circular, Elliptical, Highly Elliptical
- **Altitude Category:** LEO, MEO, GEO, Beyond GEO

**Final Feature Count:** 19 features (including engineered features)

### 7. Model Building & Evaluation

**Data Split:**
- Training: 10,065 samples (70%)
- Validation: 2,151 samples (15%)
- Test: 2,156 samples (15%)
- Stratified sampling to maintain class distribution

**Class Imbalance Handling:**
- Detected significant imbalance (ratio: 5.8:1)
- Applied SMOTE (Synthetic Minority Over-sampling Technique) when available
- Used class weights in models

**Models Trained:**

#### 1. Logistic Regression (Baseline)
- **Validation Accuracy:** 81.40%
- **Weighted F1-Score:** 79%
- Simple, interpretable baseline model

#### 2. Random Forest Classifier ⭐ (Best Model)
- **Validation Accuracy:** 92.98%
- **Test Accuracy:** 93.65%
- **Weighted F1-Score:** 92-93%
- **Configuration:**
  - n_estimators: 100
  - max_depth: 10
  - Random state: 42

#### 3. XGBoost Classifier
- Not available (library not installed)

**Advanced Evaluation Techniques:**

1. **5-Fold Cross-Validation**
   - Random Forest CV Score: ~92% (±1.5%)
   - Demonstrates model stability and consistency

2. **Learning Curves**
   - Analyzed bias-variance tradeoff
   - Assessed overfitting/underfitting
   - Training and validation curves converge at high performance

3. **Feature Importance Analysis**
   - Top features identified by Random Forest
   - Orbital parameters (MEAN_MOTION, PERIOD) most influential

4. **Confusion Matrix Analysis**
   - Per-class performance evaluation
   - Identified classes with lower recall (TBA: 14-24%)

## 📈 Model Performance

### Final Test Set Results (Random Forest)

| Metric | Score |
|--------|-------|
| **Accuracy** | **93.65%** |
| **Precision (weighted)** | 94% |
| **Recall (weighted)** | 94% |
| **F1-Score (weighted)** | 93% |

### Per-Class Performance

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| DEBRIS (0) | 0.95 | 0.98 | 0.96 | 1,265 |
| PAYLOAD (1) | 0.92 | 0.96 | 0.94 | 742 |
| ROCKET BODY (2) | 0.90 | 0.55 | 0.69 | 112 |
| TBA (3) | 1.00 | 0.24 | 0.39 | 37 |

**Key Observations:**
- ✅ Excellent performance on majority classes (DEBRIS, PAYLOAD)
- ⚠️ Lower recall on minority classes (ROCKET BODY, TBA) due to class imbalance
- 🎯 Overall model is production-ready with 93.65% accuracy

## 💻 Technologies Used

**Core Libraries:**
- Python 3.8+
- Pandas - Data manipulation
- NumPy - Numerical computations
- Matplotlib & Seaborn - Data visualization

**Machine Learning:**
- Scikit-learn - ML algorithms and preprocessing
- Imbalanced-learn (imblearn) - SMOTE implementation

**Modeling Algorithms:**
- Logistic Regression
- Random Forest Classifier
- XGBoost (optional)

## 🚀 Installation

### Prerequisites
```bash
Python 3.8 or higher
pip package manager
```

### Setup

1. Clone the repository:
```bash
git clone https://github.com/thenithin342/Space-decay-prediction-machine-learning-project.git
cd Space-decay-prediction-machine-learning-project
```

2. Install required packages:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn jupyter
```

3. Optional (for XGBoost):
```bash
pip install xgboost
```

4. Launch Jupyter Notebook:
```bash
jupyter notebook space.ipynb
```

## 📖 Usage

### Using the Trained Model

The project includes saved model files ready for predictions:

```python
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the trained model
with open('best_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load the scaler
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Prepare new data (example)
new_data = pd.DataFrame({
    'MEAN_MOTION': [12.96],
    'ECCENTRICITY': [0.0026],
    'INCLINATION': [90.27],
    'RA_OF_ASC_NODE': [325.72],
    'ARG_OF_PERICENTER': [182.04],
    'MEAN_ANOMALY': [178.06],
    'BSTAR': [0.0017],
    'MEAN_MOTION_DOT': [5.19e-06],
    'SEMIMAJOR_AXIS': [7652.14],
    'PERIOD': [111.03],
    'APOAPSIS': [1294.50],
    'PERIAPSIS': [1253.52],
    'REV_AT_EPOCH': [36475]
})

# Scale and predict
new_data_scaled = scaler.transform(new_data)
prediction = model.predict(new_data_scaled)

print(f"Predicted class: {prediction[0]}")
# 0: DEBRIS, 1: PAYLOAD, 2: ROCKET BODY, 3: TBA
```

## 📁 Project Structure

```
Space-decay-prediction-machine-learning-project/
│
├── space.ipynb              # Main Jupyter notebook with complete analysis
├── space_backup.ipynb       # Backup notebook
├── space_decay.csv          # Dataset (14,372 orbital records)
├── best_model.pkl           # Trained Random Forest model
├── scaler.pkl              # Fitted StandardScaler
├── README.md               # Project documentation
└── .gitignore              # Git ignore file
```

## 🎓 Key Learnings

### Best Practices Implemented

1. **Data Cleaning:**
   - Systematic approach to missing values
   - Domain-aware outlier handling
   - Duplicate detection and removal

2. **Preprocessing:**
   - Proper train-test splitting before scaling
   - StandardScaler for feature normalization
   - Label encoding for categorical variables

3. **Modeling:**
   - Started with simple baseline (Logistic Regression)
   - Progressive complexity (Random Forest)
   - Cross-validation for robust evaluation

4. **Evaluation:**
   - Multiple metrics (Accuracy, Precision, Recall, F1)
   - Confusion matrix analysis
   - Per-class performance evaluation
   - Learning curves for bias-variance analysis

### Common Pitfalls Avoided

✅ **No Data Leakage:** Fitted scaler only on training data  
✅ **Class Imbalance Awareness:** Applied SMOTE and class weights  
✅ **Proper Validation:** Used stratified K-fold cross-validation  
✅ **Feature Scaling:** Scaled features for distance-based algorithms  
✅ **Overfitting Prevention:** Monitored train vs validation performance  

## 🔮 Future Improvements

- [ ] Hyperparameter tuning with GridSearchCV/RandomizedSearchCV
- [ ] Ensemble methods (stacking, voting classifiers)
- [ ] Deep learning approaches (Neural Networks)
- [ ] Feature engineering based on domain knowledge
- [ ] Time-series analysis of orbital decay patterns
- [ ] Web application for real-time predictions

## 📄 License

This project is licensed under the MIT License.

## 👤 Author

**thenithin342**
- GitHub: [@thenithin342](https://github.com/thenithin342)

## 🙏 Acknowledgments

- Dataset: Space debris orbital data from satellite tracking systems
- Scikit-learn documentation and community
- Data science best practices from industry experts

---

**Note:** This project demonstrates a complete machine learning workflow from data cleaning to model deployment, suitable for learning purposes and real-world applications in space debris tracking and classification.

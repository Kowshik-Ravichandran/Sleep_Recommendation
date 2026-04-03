# 🌙 AI Sleep Lab

**Advanced Sleep Quality & Health Analysis System for Shift Workers**

An intelligent, research-grade application that uses **Machine Learning** and **Chronobiology** to analyze sleep patterns, predict sleep quality risks, and provide actionable insights — specifically designed for understanding the impact of shift work on circadian rhythms.

---

## 📌 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Machine Learning Models](#machine-learning-models)
- [Dataset](#dataset)
- [Installation & Setup](#installation--setup)
- [How to Run](#how-to-run)
- [Application Pages](#application-pages)
- [Modules Overview](#modules-overview)
- [Research Methodology](#research-methodology)
- [Screenshots](#screenshots)
- [Author](#author)

---

## Overview

Shift work often disrupts the circadian rhythm, leading to **Shift Work Sleep Disorder (SWSD)**, metabolic issues, and decreased performance — especially in healthcare professionals. This application leverages an ensemble of advanced ML models combined with circadian rhythm analysis to:

- **Analyze** sleep patterns across different demographics, shift types, and chronotypes
- **Predict** the risk of poor sleep quality using the BSQI (Behavioral Sleep Quality Index) score
- **Explain** predictions through SHAP-based Explainable AI (XAI)
- **Optimize** daily schedules to minimize health risks using an AI schedule optimizer

---

## Key Features

| Feature | Description |
|---------|-------------|
| 🧠 **Ensemble ML Prediction** | Combines XGBoost, LightGBM, Random Forest, and a PyTorch Neural Network (MLP) for accurate sleep quality prediction |
| 📊 **Interactive Visual Analytics** | Dashboards built with Plotly covering demographics, shift work, chronotype, and chrononutrition analysis |
| 🔬 **Explainable AI (XAI)** | SHAP-based feature importance, beeswarm plots, dependence plots, and force plots for model interpretability |
| ⏰ **Circadian Feature Engineering** | Novel features including Social Jetlag, Phase Delay Index, Circadian Misalignment Score, and Cosinor Rhythm Analysis |
| 📈 **Publication-Grade Statistics** | ANOVA, T-Tests with Cohen's d and η² effect sizes, Bonferroni and Benjamini-Hochberg corrections, Friedman and Wilcoxon model comparison tests |
| ⚡ **AI Schedule Optimizer** | Randomized search optimization to find lifestyle adjustments that improve sleep quality |
| 🎨 **Premium UI** | Dark-themed, glassmorphism-styled Streamlit interface with custom CSS theming |

---

## Tech Stack

| Category | Technologies |
|----------|-------------|
| **Language** | Python 3.x |
| **Web Framework** | Streamlit |
| **ML / Deep Learning** | XGBoost, LightGBM, scikit-learn, PyTorch |
| **Explainability** | SHAP |
| **Visualization** | Plotly, Matplotlib |
| **Data Processing** | Pandas, NumPy, SciPy |
| **Statistics** | SciPy (ANOVA, T-Tests, Friedman, Wilcoxon) |
| **Data Source** | Excel (openpyxl) |

---

## Project Structure

```
AI-Sleep-Lab/
│
├── Home.py                              # Main entry point (Streamlit home page)
├── README.md                            # Project documentation
├── .gitignore                           # Git ignore rules
│
├── pages/                               # Streamlit multi-page app
│   ├── 1_🌙_Prediction_Model.py        # Sleep quality prediction with ensemble models
│   ├── 2_📊_Visualization.py           # Interactive analytics dashboard
│   └── 3_🔬_Research_Analysis.py       # Publication-ready research analysis
│
├── modules/                             # Core Python modules
│   ├── __init__.py
│   ├── data_loader.py                   # Data loading & preprocessing
│   ├── prediction_engine.py             # Ensemble prediction logic
│   ├── visualization.py                 # Plotly chart generators
│   ├── statistics.py                    # Statistical tests (ANOVA, T-Tests)
│   ├── evaluation.py                    # Cross-validation & model evaluation
│   ├── circadian_analysis.py            # Circadian rhythm feature engineering
│   ├── interpretability.py              # SHAP-based model interpretability
│   ├── optimizer.py                     # AI schedule optimizer
│   ├── insights.py                      # Automated insight generation
│   └── theme.py                         # Premium UI theme & custom CSS
│
├── data/
│   └── DATA_COMPILATION.xlsx            # Primary dataset (multi-sheet Excel)
│
├── project_models/                      # Pre-trained model artifacts
│   ├── xgboost_bsq_model.pkl           # XGBoost regressor
│   ├── lightgbm_bsq_model.pkl          # LightGBM regressor
│   ├── random_forest_bsq_model.pkl     # Random Forest regressor
│   ├── mlp_bsq_model.pth              # PyTorch Neural Network (MLP)
│   ├── mlp_scaler.pkl                  # StandardScaler for MLP features
│   └── mlp_feature_names.pkl           # Feature name mapping for MLP
│
└── artifacts/
    └── shap_plots/                      # Generated SHAP analysis plots
```

---

## Machine Learning Models

### Ensemble Architecture

The system uses an **ensemble of four models** that are independently trained and their predictions averaged for the final BSQI score:

| Model | Type | Input Features | Description |
|-------|------|----------------|-------------|
| **XGBoost** | Gradient Boosting | 33 features (one-hot encoded) | Primary model with strong tabular performance |
| **LightGBM** | Gradient Boosting | 33 features (one-hot encoded) | Fast, efficient gradient boosting |
| **Random Forest** | Bagging Ensemble | 33 features (one-hot encoded) | Robust to overfitting, handles noise well |
| **MLP (Neural Network)** | Deep Learning (PyTorch) | 46 features (separate encoding) | 5-layer network with BatchNorm, LeakyReLU, and Dropout |

### MLP Architecture

```
Input (46) → Linear(256) → BatchNorm → LeakyReLU → Dropout(0.3)
          → Linear(128) → BatchNorm → LeakyReLU → Dropout(0.3)
          → Linear(64)  → BatchNorm → LeakyReLU → Dropout(0.3)
          → Linear(32)  → BatchNorm → LeakyReLU → Dropout(0.3)
          → Linear(1)   → Sigmoid → BSQI Score (0.0 – 1.0)
```

### Prediction Output

The BSQI (Behavioral Sleep Quality Index) score is interpreted as:

| Score Range | Label | Meaning |
|-------------|-------|---------|
| **0.00 – 0.49** | ⚠️ Poor Sleep | High risk of sleep-related health issues |
| **0.50 – 0.74** | ⚖️ Moderate Sleep | Acceptable but room for improvement |
| **0.75 – 1.00** | ✨ Good Sleep | Healthy sleep patterns |

---

## Dataset

The project uses the `DATA_COMPILATION.xlsx` dataset containing the following key variables:

| Category | Features |
|----------|----------|
| **Demographics** | Subject_ID, Gender, Age, Height, Weight, BMI, Ethnicity, Marital_Status, Household_Income, Highest_Education |
| **Work** | Industry, Shift_Rotation, Part-time |
| **Health** | Disease, Smoking/Vaping |
| **Chronotype** | Chronotype_Shift, Chronotype_MEQ, ScoreMEQ |
| **Sleep** | Bedtime, Sleep_latency, Actual_sleep_hours, Wake_up_time |
| **PSQI** | PSQI1–PSQI7 subscales, Total_scorePSQI, Total_sleep_quality |
| **Nutrition** | Breakfast_skipping, Largest_mealtime, Last_eating_workday, Last_eating_freeday, Eating_window_workday, Eating_window_freeday |

---

## Installation & Setup

### Prerequisites

- **Python 3.8+** installed on your system
- **pip** package manager

### Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/Kowshik-Ravichandran/Sleep_Recommendation.git
   cd Sleep_Recommendation
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate        # macOS/Linux
   # venv\Scripts\activate         # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install streamlit pandas numpy scipy scikit-learn xgboost lightgbm torch joblib plotly matplotlib shap openpyxl
   ```

4. **Verify the dataset** is placed at:
   ```
   data/DATA_COMPILATION.xlsx
   ```

5. **Verify model files** exist in:
   ```
   project_models/
   ```

---

## How to Run

Start the Streamlit application:

```bash
streamlit run Home.py
```

The app will open in your default browser at `http://localhost:8501`.

---

## Application Pages

### 🏠 Home Page (`Home.py`)
The landing page provides an overview of the project, its problem statement, proposed solution, and key features. It serves as the navigation hub for the three main modules.

### 🌙 Prediction Model (`pages/1_🌙_Prediction_Model.py`)
An interactive form where users input personal, sleep, chronotype, and nutrition data to receive a real-time BSQI sleep quality prediction. Users can:
- Select a specific model or use the full ensemble
- View individual model scores side-by-side
- Get color-coded risk labels (Poor / Moderate / Good)

### 📊 Visualization Dashboard (`pages/2_📊_Visualization.py`)
A comprehensive analytics dashboard with four analysis sections:
- **A. Sociodemographic Insights** — Sleep quality by gender, age distribution, BMI categories, education
- **B. Shift-Work & Sleep Patterns** — Sleep hours by shift type, bedtime heatmaps, latency analysis
- **C. Chronotype Analysis** — Chronotype distribution, MEQ scores, MSFSC correlations
- **D. Chrononutrition Behavior** — Meal timing vs PSQI, eating windows, workday vs. freeday patterns

Also includes tabs for **AI Insights** (automated findings), **Statistics** (ANOVA/T-Tests), and **Data View**.

### 🔬 Research Analysis (`pages/3_🔬_Research_Analysis.py`)
Publication-ready analysis module with four sections:
- **Model Performance (CV)** — 10-fold Stratified Cross-Validation with 95% Confidence Intervals
- **SHAP Interpretability** — Feature importance via SHAP (beeswarm, bar, dependence, force plots)
- **Circadian Features** — Novel engineered features (Social Jetlag, Phase Delay Index, Circadian Misalignment)
- **Statistical Tests** — T-Tests and ANOVA with Cohen's d, η², and significance stars

---

## Modules Overview

| Module | Purpose |
|--------|---------|
| `data_loader.py` | Loads data from CSV/Excel, handles multi-sheet merging, validates required columns, basic type coercion |
| `prediction_engine.py` | Loads all 4 models, preprocesses user inputs to canonical feature format, runs ensemble inference |
| `visualization.py` | Generates Plotly charts for demographics, shift work, chronotype, and nutrition analysis |
| `statistics.py` | Performs T-Tests & ANOVA with effect sizes (Cohen's d, η²), correlation heatmaps, significance stars |
| `evaluation.py` | Stratified K-Fold CV, confidence intervals, Friedman/Wilcoxon model comparison, Bonferroni/BH FDR correction |
| `circadian_analysis.py` | Circadian feature engineering — Social Jetlag, Mid-Sleep, Phase Delay, Circadian Misalignment, Cosinor analysis |
| `interpretability.py` | SHAP explainer generation, summary/bar/dependence/force plots, circadian feature grid, top feature ranking |
| `optimizer.py` | AI schedule optimizer — randomized search over actionable lifestyle parameters to minimize poor sleep risk |
| `insights.py` | Automated insight generation — detects key correlations, worst shifts, at-risk chronotypes, nutrition impacts |
| `theme.py` | Premium dark theme with glassmorphism CSS, gradient accents, animations, and responsive styling |

---

## Research Methodology

This project follows publication-grade research standards:

### Statistical Tests
- **Independent Samples T-Test** — Comparing two groups (e.g., Male vs Female sleep quality)
- **One-Way ANOVA** — Comparing 3+ groups (e.g., sleep across shift types)
- **Effect Sizes** — Cohen's d (T-Test) and Eta-squared η² (ANOVA) for practical significance
- **Multiple Testing Correction** — Bonferroni and Benjamini-Hochberg FDR

### Model Evaluation
- **10-Fold Stratified Cross-Validation** — Ensures balanced class representation
- **95% Confidence Intervals** — Using t-distribution for small sample sizes
- **Model Comparison** — Friedman test (3+ models) and Wilcoxon signed-rank (pairwise)

### Circadian Analysis
- **Social Jetlag** — Difference between mid-sleep on workdays vs. free days
- **Phase Delay Index** — Deviation from a 10 PM bedtime reference
- **Circadian Misalignment** — Mismatch between chronotype and actual sleep schedule, amplified by shift type
- **Cosinor Analysis** — Fitting cosine curves to model circadian rhythmicity (Mesor, Amplitude, Acrophase)
- **Chronotype-Shift Match** — Binary indicator of whether circadian preference aligns with work schedule

---

## Screenshots

> Run the app with `streamlit run Home.py` and explore the interactive dashboards.

---

## Author

**Kowshik**

- Built with ❤️ using Streamlit & Python
- Powered by Deep Learning Ensemble

---

## License

This project is for academic and educational purposes.

---

*© 2024 Kowshik — AI Sleep Lab v2.0*

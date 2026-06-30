# Claude Code Specifications

This document lists all Claude Code specifications used to build this project.

---

## Week 1-2: Data & EDA

### Spec 1: Data Generation

> Generate 50,000 synthetic customer records for a churn prediction dataset with 30 features. Include demographic features (age, tenure, contract type), usage features (login frequency, feature adoption, days since last login), financial features (monthly spend, plan tier), and a sentiment score. Use a logistic formula to assign churn probability so that roughly 13–14% of customers churn. Save as `synthetic_churn_data.csv`.

### Spec 2: EDA Analysis

> Run a 7-section EDA analysis on the synthetic churn dataset. Cover: (1) dataset overview and shape, (2) churn rate and class imbalance, (3) numerical feature distributions by churn class with Welch t-tests, (4) categorical feature breakdowns, (5) Pearson correlation heatmap, (6) at-risk rule based on engagement score flagging, (7) key insights summary. Print all statistics to console.

### Spec 3: Visualizations

> Create 5 matplotlib/seaborn visualizations saved to an `outputs/` folder at 300 DPI. Use green (#2CA02C) for retained and red (#D62728) for churned customers. Charts: (1) churn rate bar chart, (2) feature distributions by churn class, (3) correlation heatmap, (4) contract type vs churn stacked bar, (5) churn rate by engagement decile using `pd.qcut` with q=10.

---

## Week 3-4: Features & Engineering

### Spec 4: Feature Engineering Pipeline

> Build a feature engineering pipeline that transforms 29 raw features into 61 ML-ready features. Steps: (1) one-hot encode categoricals, (2) create binary flag features (high_spender, long_tenure, at_risk), (3) create ratio features (spend_per_login, adoption_rate), (4) create composite scores: `churn_risk_score` and `loyalty_score`, (5) create `sentiment_momentum = (sentiment_score/100 + sentiment_trend*0.2).clip(0,1)`, (6) detect and drop collinear features with Pearson r > 0.95. Apply StandardScaler. Save the feature matrix as `synthetic_churn_features.csv`.

---

## Week 5: Models

### Spec 5: Model Training & Comparison

> Train and compare 4 models on the engineered feature set using a stratified 70/15/15 train/val/test split and StratifiedKFold CV (k=5). Models: (1) Logistic Regression with GridSearchCV over C=[0.001, 0.01, 0.1, 1, 10] and penalty=[l1, l2], class_weight='balanced'; (2) XGBoost on structured features only (42 cols), GridSearchCV over max_depth=[3,5,7], learning_rate=[0.01,0.1,0.2], n_estimators=[200], scale_pos_weight=6.46; (3) XGBoost on sentiment features only (19 cols); (4) Stacking ensemble using OOF predictions from models 1–3 as inputs to a meta LogisticRegression. Fit StandardScaler on X_train only to avoid leakage. Report AUC-ROC on the test set for all models. Save the best model as `best_model.pkl`.

### Spec 6: Threshold Optimization & Cost Analysis

> Load `best_model.pkl` and run threshold optimization using a cost matrix: TP benefit = $5,000 (customer saved), FP cost = -$500 (wasted intervention). Sweep 500 thresholds from 0 to 1. Find and report 3 optimal thresholds: (1) Youden's J (max TPR - FPR), (2) F1-optimal, (3) Cost-optimal (max net value). Plot 5 charts: ROC curve, precision-recall curve, net value vs threshold, confusion matrices at each optimal threshold, and cost breakdown.

### Spec 7: XGBoost on All 61 Features vs Best LR

> Deploy a new XGBoost model trained on all 61 engineered features (not just structured or sentiment subsets). Use the same train/val/test split. Run GridSearchCV with the same hyperparameter grid. Compare AUC-ROC and net business value at the cost-optimal threshold against the best Logistic Regression model from Spec 5. Generate 5 comparison charts saved as `xgb_full_*.png`.

---

## Week 6: A/B Testing & Business Impact

### Spec 8: A/B Test Design & Simulation

> Design and simulate an A/B test for a churn retention intervention. Power analysis: baseline churn rate = 13.47%, MDE = 2% absolute reduction, alpha = 0.05, target power = 80%, relative lift = 25%. Calculate required sample size per group using the two-proportion z-test formula. Then simulate 1,000 A/B tests using binomial draws at the true effect size (3.37% absolute). Use chi-square test for significance in each simulation. Report: mean lift, 95% confidence interval, empirical power, and significance rate. Plot 4 charts: power curve, simulated lift distribution, significance rate vs effect size, sample size vs MDE. Save as `ab1.png`–`ab4.png`.

### Spec 9: Business Impact & Deployment Strategy

> Calculate business impact using 5-year LTV (not annual revenue) as the benefit metric. Base case: intervention deployed on model's predicted churners at cost-optimal threshold. Compute: customers saved, LTV recovered, total intervention cost, net benefit, and ROI. Build a sensitivity heatmap across 5 effect sizes × 5 intervention costs. Design a 3-phase deployment plan: (1) Canary (5% traffic), (2) Expansion (25%), (3) Full rollout (100%). Plot 5 charts including ROI waterfall, sensitivity heatmap, and phased deployment timeline. Save as `biz1.png`–`biz5.png`.

### Spec 10: Project Report

> Update the HTML project report to cover all 8 pipeline steps with stat boxes, results tables, code blocks, and callout boxes (insight / warning / success / danger). Include a sticky navigation bar, 8 content sections, a pitfalls section documenting 10 common ML mistakes encountered and fixed (e.g., data leakage, accuracy trap, LTV vs revenue confusion), and a summary section. Report should be self-contained in a single HTML file with embedded CSS.

---

## Key Insight

Each spec was written to be clear and specific — defining the exact inputs, outputs, metrics, and file names expected. Claude Code generated production-quality Python that was reviewed, tested, and iterated on. Key iteration cycles included:

- **Data leakage fix**: StandardScaler was initially fit on all 50k rows; corrected to fit on training data only
- **ROI calculation fix**: Initial model used annual revenue (giving -66% ROI); corrected to use 5-year LTV (giving +68.4% ROI)
- **NumPy compatibility fix**: `np.trapz()` deprecated in NumPy 2.0; replaced with `np.trapezoid()`
- **Model surprise**: Logistic Regression (AUC 0.8386) outperformed XGBoost (AUC 0.8310) because feature engineering pre-linearized non-linear patterns, removing XGBoost's main advantage

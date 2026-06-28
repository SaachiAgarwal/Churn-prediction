# Customer Churn Prediction

End-to-end machine learning project: synthetic data generation → exploratory analysis → feature engineering → model training → threshold optimisation → A/B test design → business impact & deployment strategy.

---

## Results at a Glance

| Metric | Value |
|---|---|
| Dataset | 50,000 synthetic customers × 30 features |
| Engineered features | 61 (from 30 raw) |
| Best model | Logistic Regression (L1, C=0.01) |
| AUC-ROC | **0.8386** |
| Deployment threshold | 0.409 (cost-optimal) |
| Recall @ threshold | 81.8% |
| A/B test sample size | 4,283 per group |
| Empirical power | 99.9% |
| Full rollout net benefit | **$2,302,528** |
| ROI (5-year LTV) | **68.4%** |

---

## Project Structure

```
Churn-prediction/
├── generate_synthetic_churn.py   # Step 1: generate 50k customer dataset
├── analyze_churn.py              # Step 2: 7-section EDA
├── visualize_churn_v2.py         # Step 2: 5 matplotlib/seaborn charts
├── feature_engineering.py        # Step 3: 29 raw → 61 ML-ready features
├── train_models.py               # Step 4: train & compare 4 models
├── threshold_analysis.py         # Step 5: ROC + cost matrix optimisation
├── train_xgb_full.py             # Step 6: XGBoost all-features vs LR
├── ab_test_simulation.py         # Step 7: power analysis + 1,000 simulations
├── business_impact.py            # Step 8: financial model + deployment plan
├── churn_project_report.html     # Full HTML report (open in browser)
├── synthetic_churn_50k.csv       # Generated dataset
├── synthetic_churn_features.csv  # Engineered feature dataset
├── best_model.pkl                # Serialised best model + scaler
├── model_comparison.csv          # All model metrics
└── outputs/                      # All charts (25 PNG files)
```

---

## Step-by-Step Walkthrough

### Step 1 — Data Generation

Generated 50,000 synthetic customers with 30 features using realistic statistical distributions. Churn labels created via a logistic formula with domain-realistic weights:

```python
logit = -1.25
  - (sentiment_score - 50) * 0.04   # low sentiment → churn
  + escalated_support * 1.8          # escalation is strong signal
  + complaint_count_6m * 0.35
  - feature_adoption_rate * 1.2      # engaged users stay
  + (contract_type == 'monthly') * 0.6
  - auto_renewal_enabled * 0.5
  + payment_failures_6m * 0.4
  + noise

p_churn = 1 / (1 + e^(-logit))
```

Intercept tuned to produce **13.4% churn rate** — realistic for SaaS.

**30 features across 5 categories:** demographics, product/plan, engagement, financial, sentiment & support.

---

### Step 2 — Exploratory Data Analysis

7-section analysis using scipy.stats, pandas, and seaborn.

**Key findings:**
- Sentiment score is the strongest predictor (Pearson r = –0.35)
- Escalated support multiplies churn **4.44×** (44% vs 10% baseline)
- 6–12 month tenure customers have the highest churn (15.7%)
- Monthly contract customers churn 2× more than annual
- 4 simple rules catch **77.6% of churners** (but with 42% false positive rate)

**5 visualisations:** class distribution, churn by product, feature correlations, sentiment vs churn, engagement score vs churn.

---

### Step 3 — Feature Engineering Pipeline

Transformed 30 raw features into 61 ML-ready features:

| Stage | Count |
|---|---|
| Raw features | 30 |
| Features built | 70 |
| Dropped (collinear r > 0.95) | 9 |
| **Final features** | **61** |

**Types of features created:**
- One-hot encoding (region, plan, product, contract, spend_trend)
- Binary flags (escalation, complaints, payment risk, spend decline)
- Ratio features (spend per product, support per tenure)
- Composite scores: `churn_risk_score`, `loyalty_score`, `engagement_score`
- 10 extended sentiment features including `sentiment_momentum`

**Key composite scores:**
```python
churn_risk_score = (
    sentiment_declining * 0.15 + escalation_flag * 0.25
    + complaint_flag * 0.15 + spend_declining * 0.20
    + engagement_decline * 0.10 + contract_risk * 0.10
    + payment_risk_flag * 0.05
)

loyalty_score = (
    contract_type_numeric * 0.20 + contract_stability * 0.15
    + mature_customer_flag * 0.15 + active_user_flag * 0.20
    + feature_adoption_flag * 0.15 + high_spender_flag * 0.15
)
```

**Pitfall fixed:** StandardScaler fit on training data only — not all 50k rows (which would be data leakage).

---

### Step 4 — Model Training & Comparison

**Data split:** 70% train / 15% val / 15% test — stratified to preserve 13.4% churn rate in each split.

**4 models trained:**

| Model | AUC-ROC | AUC-PR | F1 | Notes |
|---|---|---|---|---|
| **Logistic Regression** | **0.8386** | **0.5194** | **0.511** | Winner — L1, C=0.01 |
| Ensemble Stack | 0.8184 | 0.4971 | 0.487 | OOF stacking of XGB2+3 |
| XGBoost (Sentiment) | 0.7987 | 0.4702 | 0.461 | 19 sentiment features |
| XGBoost (Structured) | 0.7905 | 0.4612 | 0.449 | 42 structured features |

**Why Logistic Regression won:**
Feature engineering pre-computed the non-linear patterns XGBoost would have discovered itself. With composite scores already encoding the signal linearly, LR with strong L1 regularisation outperformed complex tree models.

**Class imbalance handling:**
- LR: `class_weight='balanced'`
- XGBoost: `scale_pos_weight=6.46` (non-churners / churners ratio)

**Hyperparameter tuning:** GridSearchCV with 5-fold stratified CV on training set only.

---

### Step 5 — Threshold Optimisation & Cost Matrix

**Cost structure:**
- True Positive (churner saved): +$5,000
- False Positive (non-churner contacted): –$500
- False Negative / True Negative: $0

**Three optimal thresholds:**

| Method | Threshold | Recall | Precision | Net Value |
|---|---|---|---|---|
| **Cost-Optimal** | **0.409** | **81.8%** | **29.2%** | **$3,105,500** |
| Youden's J | 0.478 | 76.4% | 32.5% | $3,056,500 |
| F1-Optimal | 0.701 | 53.4% | 49.0% | $2,401,500 |

**Recommendation:** cost-optimal threshold (0.409) — contact 2,809 customers, save 820 churners, net value $3.1M.

Low precision (29%) is acceptable: at $5,000 per churner saved vs $500 per false alarm, the economics strongly favour broad outreach.

---

### Step 6 — XGBoost (All 61 Features) vs LR

Trained XGBoost on all 61 engineered features to test whether giving it the full feature set would close the gap with LR.

**Result: LR still wins** (AUC 0.8386 vs 0.8310, net value advantage $82,000).

XGBoost would win with: raw features (no engineering), complex cross-feature interactions, much larger datasets (500k+), or more aggressive hyperparameter tuning.

---

### Step 7 — A/B Test Design & Simulation

**Purpose:** Prove the retention intervention causes churn reduction — not just correlates with it.

**Power analysis (two-proportion z-test):**

| Parameter | Value |
|---|---|
| Baseline churn | 13.47% |
| Min detectable effect (MDE) | 2.0% absolute |
| Significance level α | 0.05 |
| Target power | 80% |
| **Required n per group** | **4,283** |
| Power @ MDE | 80.0% |
| Power @ true treatment (3.37% abs) | 99.8% |

**Simulation (1,000 runs):**

```
Mean lift:               24.8%
95% CI on lift:          [15.7%, 33.3%]
% simulations p < 0.05:  99.9%  ← empirical power
```

**Statistical concepts used:**
- Two-proportion z-test with Youden-corrected sample size formula
- Chi-square test for significance in each simulation
- Binomial draws for realistic random variation
- Empirical vs theoretical power distinction

---

### Step 8 — Business Impact & Deployment

**Financial model (full rollout, 6,735 at-risk customers):**

| Line Item | Value |
|---|---|
| Total outreach cost | $3,367,500 |
| Customers saved | 227 |
| Annual revenue saved | $1,134,006 |
| 5-year LTV saved | $5,670,028 |
| **Net benefit** | **$2,302,528** |
| **ROI (LTV basis)** | **68.4%** |
| Payback period | 35.6 months |

ROI uses 5-year LTV (not year-1 revenue) because the intervention cost is one-time but the retained customer pays for 5 years.

**Break-even:** intervention needs at least **15% relative churn reduction** to be ROI-positive.

**3-Phase Deployment:**

| Phase | Weeks | Sample | Cost | Goal |
|---|---|---|---|---|
| Canary | 1–4 | 50/group | ~$25K | Test infrastructure |
| Expansion | 5–12 | 3,367/group | ~$842K | Validate ROI at scale |
| Full Rollout | 13+ | All 6,735 | ~$3.4M | Maximum impact |

**Guardrails:**
- Pause if treatment churn increases vs control
- Alert if NPS drops > 10 points
- Alert if model AUC drops below 0.75
- Annual model retraining required

---

## Key Learnings

1. **Feature engineering > model complexity.** LR on 61 engineered features beat XGBoost across every metric. Invest in features before chasing algorithms.

2. **Use business metrics, not just AUC.** Cost-optimal threshold generated $700K more than F1-optimal despite lower precision — because the cost structure is asymmetric.

3. **Threshold is a business decision.** Default 0.5 is almost never optimal. Always select threshold based on actual intervention costs and benefits.

4. **Power analysis before the experiment.** Running an A/B test without knowing if you have enough data wastes customers and budget.

5. **Phased deployment reduces risk.** Canary → expansion → full rollout lets you catch problems before committing the full budget.

---

## Pitfalls Fixed

| # | Pitfall | Fix |
|---|---|---|
| 1 | Data leakage in StandardScaler | Fit scaler on train only |
| 2 | Accuracy trap with class imbalance | Use AUC-ROC, F1, net value |
| 3 | ROI vs annual revenue (not LTV) | ROI = (LTV saved - cost) / cost |
| 4 | Class imbalance ignored | class_weight='balanced', scale_pos_weight |
| 5 | Default threshold 0.5 | Cost-optimal threshold 0.409 |
| 6 | Feature subset splitting (not validated) | Tested empirically — LR won |
| 7 | np.trapz removed in NumPy 2.0 | Replaced with np.trapezoid() |
| 8 | Collinear features left in | Dropped 9 features with r > 0.95 |

---

## Running the Project

```bash
# Install dependencies
pip install pandas numpy scikit-learn xgboost matplotlib seaborn scipy

# Run in order
python generate_synthetic_churn.py   # ~10s
python analyze_churn.py              # ~15s
python visualize_churn_v2.py         # ~10s
python feature_engineering.py        # ~20s
python train_models.py               # ~3-5min (GridSearchCV)
python threshold_analysis.py         # ~30s
python train_xgb_full.py             # ~1-2min
python ab_test_simulation.py         # ~60s
python business_impact.py            # ~10s
```

Open `churn_project_report.html` in a browser for the full interactive report.

---

## Tech Stack

| Tool | Use |
|---|---|
| Python 3.10+ | Core language |
| pandas / numpy | Data manipulation |
| scikit-learn | LR, GridSearchCV, metrics, preprocessing |
| XGBoost | Gradient boosted trees |
| matplotlib / seaborn | All visualisations |
| scipy.stats | EDA, chi-square test, power analysis |

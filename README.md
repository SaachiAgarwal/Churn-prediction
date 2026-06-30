# Customer Churn Prediction — End-to-End ML Project

> **A SaaS company loses $2.3M+ annually to preventable churn.**
> This project builds a production-ready ML system that identifies at-risk customers with 81.8% recall, optimises intervention spend using a cost matrix, validates impact through A/B test simulation, and delivers a 68.4% ROI on 5-year LTV — with a 3-phase deployment plan ready to ship.

---

## 🔑 Key Findings

- **Logistic Regression beat XGBoost** — because feature engineering pre-linearised the signal. Invest in features before chasing algorithms.
- **Default threshold 0.5 leaves $700K on the table** — cost-optimal threshold (0.409) generated $3.1M vs $2.4M at F1-optimal.
- **Low precision (29%) is the right answer** — when saving a churner is worth $5K and a false alarm costs $500, broad outreach is economically correct.
- **99.9% empirical A/B test power** — at the true treatment effect (3.37% absolute lift), the intervention is statistically unambiguous.
- **Break-even at 15% relative churn reduction** — the business case is robust even under conservative assumptions.

---

## 📊 Project Overview

End-to-end ML pipeline covering every stage from raw data to business deployment:

| Stage | What Was Built |
|---|---|
| Data Generation | 50,000 synthetic customers, 30 features, 13.4% churn rate |
| EDA | 7-section analysis — correlations, t-tests, at-risk rules |
| Feature Engineering | 29 raw → 61 ML-ready features, collinearity removal |
| Model Training | 4 models: LR, XGB (structured), XGB (sentiment), Ensemble |
| Threshold Optimisation | 3 optimal thresholds via cost matrix, Youden's J, F1 |
| Model Comparison | XGBoost all-features vs best LR — LR wins |
| A/B Test Design | Power analysis + 1,000 simulations |
| Business Impact | Financial model, sensitivity heatmap, 3-phase deployment |

**Best model:** Logistic Regression (L1, C=0.01) — AUC-ROC **0.8386**, Net Value **$3,105,500**, ROI **68.4%**

---

## 📅 6-Week Roadmap

### Week 1 — Data Foundation
- Generated 50,000 synthetic customers with realistic statistical distributions
- Used a logistic formula with domain-realistic feature weights to assign churn labels
- Tuned intercept to produce 13.4% churn rate — representative of SaaS benchmarks

### Week 2 — Exploratory Analysis
- 7-section EDA: class imbalance, feature distributions, Pearson correlations, Welch t-tests
- Key discovery: sentiment score is the strongest predictor (r = –0.35)
- Escalated support multiplies churn 4.44× (44% vs 10% baseline)
- Built 4-rule at-risk system catching 77.6% of churners

### Week 3 — Feature Engineering
- Expanded 29 raw features to 61 ML-ready features
- Created composite scores: `churn_risk_score`, `loyalty_score`, `sentiment_momentum`
- Dropped 9 collinear features (Pearson r > 0.95)
- Fixed data leakage: StandardScaler fit on training data only

### Week 4 — Model Training & Comparison
- Trained 4 models with GridSearchCV and 5-fold stratified CV
- Handled class imbalance: `class_weight='balanced'` (LR), `scale_pos_weight=6.46` (XGB)
- LR outperformed XGBoost — feature engineering removed XGBoost's main advantage
- Stacking ensemble (0.818) still lost to LR (0.839)

### Week 5 — Threshold & Cost Optimisation
- Swept 500 thresholds across ROC curve
- Applied cost matrix: TP = +$5,000, FP = –$500
- Found cost-optimal threshold (0.409) generates $700K more than F1-optimal
- Validated: XGBoost on all 61 features still loses to LR by $82K net value

### Week 6 — A/B Testing & Business Impact
- Designed statistically rigorous A/B test: n=4,283 per group, 80% power at 2% MDE
- Simulated 1,000 trials — 99.9% empirical power at true treatment effect
- Built full financial model on 5-year LTV basis (not annual revenue)
- Designed 3-phase deployment: Canary → Expansion → Full Rollout

---

## 💡 Why This Project

Most churn tutorials stop at model accuracy. This project was built to answer the questions that actually matter in production:

- **Which threshold should we deploy?** Not 0.5 — the cost structure determines it.
- **How do we prove causation, not just correlation?** A/B test with proper power analysis.
- **What's the actual financial return?** ROI on LTV, not on year-1 revenue.
- **How do we roll out safely?** Phased deployment with guardrails and kill switches.
- **What could go wrong?** 10 pitfalls documented and fixed across the pipeline.

The goal was to build something a Data Scientist would be proud to present to a CFO — not just a notebook with a confusion matrix.

---

## 📊 Model Performance

### Model Comparison

| Model | AUC-ROC | AUC-PR | F1 | Notes |
|---|---|---|---|---|
| **Logistic Regression** | **0.8386** | **0.5194** | **0.511** | Winner — L1, C=0.01 |
| Ensemble Stack | 0.8184 | 0.4971 | 0.487 | OOF stacking |
| XGBoost (Sentiment) | 0.7987 | 0.4702 | 0.461 | 19 sentiment features |
| XGBoost (Structured) | 0.7905 | 0.4612 | 0.449 | 42 structured features |
| XGBoost (All 61 features) | 0.8310 | — | — | Still loses to LR |

### Threshold Comparison

| Method | Threshold | Recall | Precision | Net Value |
|---|---|---|---|---|
| **Cost-Optimal** | **0.409** | **81.8%** | **29.2%** | **$3,105,500** |
| Youden's J | 0.478 | 76.4% | 32.5% | $3,056,500 |
| F1-Optimal | 0.701 | 53.4% | 49.0% | $2,401,500 |

### Business Impact (Full Rollout)

| Metric | Value |
|---|---|
| Customers flagged | 6,735 |
| Customers saved | 227 |
| Total intervention cost | $3,367,500 |
| 5-year LTV recovered | $5,670,028 |
| **Net benefit** | **$2,302,528** |
| **ROI** | **68.4%** |
| Payback period | 35.6 months |

---

## 🚀 Quick Start

### Prerequisites

```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn scipy
```

### Run the Full Pipeline

```bash
python generate_synthetic_churn.py   # ~10s  — generates 50k customer dataset
python analyze_churn.py              # ~15s  — 7-section EDA
python visualize_churn_v2.py         # ~10s  — 5 charts saved to outputs/
python feature_engineering.py        # ~20s  — 29 → 61 features
python train_models.py               # ~5min — GridSearchCV across 4 models
python threshold_analysis.py         # ~30s  — cost matrix & threshold sweep
python train_xgb_full.py             # ~2min — XGBoost all features vs LR
python ab_test_simulation.py         # ~60s  — power analysis + 1,000 simulations
python business_impact.py            # ~10s  — financial model + deployment plan
```

Open `churn_project_report.html` in a browser for the full interactive report with all findings, charts, and pitfalls.

---

## 📁 Directory Structure

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
│
├── churn_project_report.html     # Full HTML report (open in browser)
├── CLAUDE_CODE_SPECS.md          # All Claude Code specs used to build this
│
├── synthetic_churn_50k.csv       # Raw generated dataset
├── synthetic_churn_features.csv  # Engineered feature matrix (61 features)
├── best_model.pkl                # Serialised best model + scaler
├── model_comparison.csv          # All model metrics
│
└── outputs/                      # All charts (25 PNG files)
    ├── viz1.png – viz5.png       # EDA visualisations
    ├── t1.png – t5.png           # Threshold analysis charts
    ├── xgb_full_*.png            # XGBoost vs LR comparison
    ├── ab1.png – ab4.png         # A/B test charts
    └── biz1.png – biz5.png       # Business impact charts
```

---

## ⚠️ Limitations

**Data**
- Dataset is synthetic — real customer data would have messier distributions, missing values, and domain-specific signals not captured here
- Churn labels were generated from a known formula; real churn is harder to model because the true drivers are unknown

**Model**
- LR winning over XGBoost is partly an artefact of the feature engineering encoding the non-linearities explicitly. With raw features, XGBoost would likely win
- No temporal validation — a time-based train/test split would better simulate production conditions
- Model drift not addressed — real deployments need retraining schedules and drift detection

**Business Case**
- $5,000 LTV and intervention costs are illustrative assumptions; real ROI depends heavily on actual unit economics
- A/B test simulation used synthetic data — real-world experiments introduce confounders (seasonality, selection bias, network effects)
- 5-year LTV assumes constant retention after intervention, which may be optimistic

**Deployment**
- No serving infrastructure built — model outputs are offline batch scores, not a real-time API
- No monitoring or alerting implementation — the guardrails described are design recommendations only

---

## 📚 Key Insights

1. **Feature engineering beats model complexity.** LR on 61 engineered features beat XGBoost on every metric. The right features matter more than the right algorithm.

2. **The accuracy trap is real.** Predicting "no churn" for everyone gives 86.6% accuracy. Always evaluate on AUC-ROC, F1, and business value — not accuracy.

3. **Threshold is a business decision, not a modelling one.** Default 0.5 is almost never optimal. Set threshold using the actual cost structure of your intervention.

4. **ROI must use LTV, not annual revenue.** Intervention cost is one-time; retained customer revenue compounds for years. Using year-1 revenue gave –66% ROI; LTV gave +68.4%.

5. **Power analysis before the experiment.** Running an underpowered A/B test wastes customers and budget. Calculate required sample size first.

6. **Empirical power ≠ theoretical power.** Theoretical power (80%) was calculated for the MDE (2% abs). Empirical power (99.9%) reflects the true treatment effect (3.37% abs) — a much easier signal to detect.

7. **Phased deployment is risk management.** Canary → expansion → full rollout lets you validate ROI at each stage before committing the full budget.

8. **Data leakage is subtle.** Fitting StandardScaler on all 50k rows before the train/test split is leakage — it passes test-set information into training. Always fit transformers on training data only.

---

## 📧 Contact

**Saachi Agarwal**
[LinkedIn →](https://www.linkedin.com/in/saachi-ag/)

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

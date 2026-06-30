# Model Card — Customer Churn Prediction

> Inspired by the [Model Cards for Model Reporting](https://arxiv.org/abs/1810.03993) framework (Mitchell et al., 2019).

---

## Model Details

| Field | Value |
|---|---|
| **Model name** | Customer Churn Classifier |
| **Version** | 1.0 |
| **Type** | Binary classification (churn / retained) |
| **Algorithm** | Logistic Regression with L1 regularisation |
| **Framework** | scikit-learn 1.x |
| **Serialised artifact** | `best_model.pkl` |
| **Developed by** | Saachi Agarwal |
| **Contact** | [linkedin.com/in/saachi-ag](https://www.linkedin.com/in/saachi-ag/) |
| **Last updated** | June 2026 |

### Hyperparameters

| Parameter | Value |
|---|---|
| `penalty` | l1 |
| `C` (inverse regularisation) | 0.01 |
| `solver` | liblinear |
| `class_weight` | balanced |
| `max_iter` | 1000 |

---

## Intended Use

### Primary Use Case
Score SaaS customers by churn probability so the retention team can prioritise outreach before customers cancel.

### Intended Users
- Customer success teams allocating outreach budget
- Data scientists maintaining and retraining the model
- Product teams designing retention interventions

### Out-of-Scope Uses
- **Do not use** to make automated cancellation or pricing decisions without human review
- **Do not use** on non-SaaS customer populations — the feature set was designed for subscription businesses
- **Do not use** as the sole signal for high-stakes personalisation (e.g. discounts, contract changes) without A/B validation

---

## Training Data

| Property | Value |
|---|---|
| Source | Synthetic — generated via `generate_synthetic_churn.py` |
| Size | 50,000 customers |
| Churn rate | 13.4% (6,700 churners / 43,300 retained) |
| Train split | 35,000 rows (70%, stratified) |
| Validation split | 7,500 rows (15%, stratified) |
| Test split | 7,500 rows (15%, stratified) |

### Feature Summary

| Category | Raw Features | Engineered Features |
|---|---|---|
| Demographics | age, region, company size | one-hot encodings, tenure flags |
| Product & Plan | plan tier, contract type, product line | contract_risk, contract_stability |
| Engagement | login frequency, feature adoption, days since login | engagement_score, active_user_flag |
| Financial | monthly spend, payment failures, spend trend | spend_per_product, high_spender_flag |
| Sentiment & Support | sentiment score, sentiment trend, escalation, complaints | sentiment_momentum, churn_risk_score, loyalty_score |
| **Total** | **29 raw** | **61 ML-ready (after dropping 9 collinear)** |

### Key Composite Features

```python
churn_risk_score = (
    sentiment_declining * 0.15 + escalation_flag * 0.25
    + complaint_flag * 0.15   + spend_declining * 0.20
    + engagement_decline * 0.10 + contract_risk * 0.10
    + payment_risk_flag * 0.05
)

loyalty_score = (
    contract_type_numeric * 0.20 + contract_stability * 0.15
    + mature_customer_flag * 0.15 + active_user_flag * 0.20
    + feature_adoption_flag * 0.15 + high_spender_flag * 0.15
)

sentiment_momentum = (sentiment_score / 100 + sentiment_trend * 0.2).clip(0, 1)
```

---

## Evaluation

### Test Set Performance (7,500 held-out rows)

| Metric | Value |
|---|---|
| AUC-ROC | **0.8386** |
| AUC-PR | **0.5194** |
| F1 Score | **0.511** |
| Accuracy | 77.4% |

> **Note:** Accuracy is not the primary metric here due to class imbalance (13.4% churn). AUC-ROC and net business value are the decision metrics.

### Threshold Analysis (Cost Matrix)

Cost structure used: TP = +$5,000 (churner saved), FP = –$500 (wasted intervention)

| Threshold Method | Threshold | Recall | Precision | Net Value |
|---|---|---|---|---|
| **Cost-Optimal** *(deployed)* | **0.409** | **81.8%** | **29.2%** | **$3,105,500** |
| Youden's J | 0.478 | 76.4% | 32.5% | $3,056,500 |
| F1-Optimal | 0.701 | 53.4% | 49.0% | $2,401,500 |

**Deployed threshold: 0.409** — maximises net business value, not classification balance. Low precision (29.2%) is intentional given the asymmetric cost structure.

### Comparison with Alternatives

| Model | AUC-ROC | Net Value | Notes |
|---|---|---|---|
| **Logistic Regression (deployed)** | **0.8386** | **$3,105,500** | Winner |
| XGBoost (all 61 features) | 0.8310 | $3,023,500 | –$82K vs LR |
| Stacking Ensemble | 0.8184 | — | OOF meta-LR |
| XGBoost (sentiment only) | 0.7987 | — | 19 features |
| XGBoost (structured only) | 0.7905 | — | 42 features |
| Baseline (always predict no-churn) | 0.500 | $0 | Accuracy trap |

---

## Business Impact

### A/B Test Validation

| Parameter | Value |
|---|---|
| Baseline churn rate | 13.47% |
| Minimum detectable effect | 2.0% absolute |
| Required sample size | 4,283 per group |
| Significance level α | 0.05 |
| Target power | 80% |
| Empirical power (1,000 simulations) | **99.9%** |
| Mean observed lift | 24.8% relative |
| 95% CI on lift | [15.7%, 33.3%] |

### Financial Projection (Full Rollout)

| Metric | Base Case | Conservative (–30%) |
|---|---|---|
| Customers flagged | 6,735 | 6,735 |
| Customers saved | 227 | 159 |
| 5-year LTV recovered | $5,670,028 | $3,969,020 |
| Total intervention cost | $3,367,500 | $3,367,500 |
| **Net benefit** | **$2,302,528** | **$601,520** |
| **ROI** | **68.4%** | **17.9%** |
| Break-even churn reduction | 15% relative | — |

### Deployment Plan

| Phase | Timeline | Sample Size | Purpose |
|---|---|---|---|
| Canary | Weeks 1–4 | 50/group | Validate infrastructure |
| Expansion | Weeks 5–12 | 3,367/group | Validate ROI at scale |
| Full Rollout | Week 13+ | All 6,735 | Maximum impact |

---

## Limitations and Risks

### Data Limitations
- **Synthetic data** — the model was trained on generated data. Real customer behaviour has messier distributions, missing values, and unobserved confounders.
- **Static snapshot** — no temporal features (e.g. trend over last 90 days vs previous 90 days). A time-series-aware model could perform better.
- **Known churn formula** — labels were generated from a known logistic formula. In production, the true drivers of churn are unknown, which typically reduces model performance.

### Model Limitations
- **No concept drift handling** — customer behaviour changes over time; model AUC will degrade without retraining.
- **No uncertainty estimates** — the model outputs a point probability, not a confidence interval. High-uncertainty predictions (near the 0.409 threshold) should be treated with caution.
- **Feature leakage risk in production** — features like `sentiment_score` and `escalation_flag` require real-time data pipelines; stale data will degrade performance.

### Fairness Considerations
- The model includes `region` and `company_size` as features. These should be monitored to ensure churn scores are not systematically biased against any customer segment.
- Intervention outreach should be reviewed to confirm it does not disproportionately target or exclude specific demographics.

### Known Failure Modes
| Scenario | Risk |
|---|---|
| New product lines added | Feature distribution shifts; retrain required |
| Pricing changes | Spend-based features lose calibration |
| Sentiment data unavailable | `sentiment_momentum` and `churn_risk_score` drop to zero; significant performance degradation expected |
| Class imbalance shifts | If churn rate moves far from 13.4%, `scale_pos_weight` and `class_weight` need recalibration |

---

## Ethical Considerations

- **Transparency:** Customers flagged as at-risk should receive genuine value-add outreach — not manipulative retention tactics.
- **Human oversight:** All retention decisions should involve a human review step, especially for high-value accounts.
- **Data minimisation:** Only features necessary for churn prediction should be collected and stored.
- **No adverse action:** This model should not be used to restrict services, change pricing, or penalise customers based on predicted churn probability.

---

## Maintenance

| Task | Frequency | Owner |
|---|---|---|
| Model retraining | Quarterly or when AUC < 0.75 | Data Science team |
| Feature pipeline validation | Monthly | Data Engineering |
| Threshold recalibration | After any pricing/cost change | Data Science + Finance |
| Fairness audit | Semi-annually | Data Science + Legal |
| A/B test for new interventions | Per campaign | Data Science + CS team |

### Guardrails (Trigger Retraining or Pause)
- AUC-ROC drops below **0.75** on a monthly holdout sample
- Churn rate drifts outside **10%–18%** in production scoring
- Treatment group churn rate exceeds control in a running A/B test
- NPS drops more than **10 points** post-intervention rollout

---

## Reproducibility

```bash
# Full pipeline — deterministic with random_state=42 everywhere
python generate_synthetic_churn.py
python feature_engineering.py
python train_models.py

# Outputs: best_model.pkl, model_comparison.csv
```

All random seeds are set via `random_state=42`. Results are fully reproducible on the same Python + library versions.

| Library | Version |
|---|---|
| Python | 3.10+ |
| scikit-learn | 1.x |
| XGBoost | 1.7+ |
| pandas | 1.5+ |
| numpy | 1.24+ / 2.0+ |
| scipy | 1.10+ |

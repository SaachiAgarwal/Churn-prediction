import numpy as np
import pandas as pd
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
N = 50_000

# Demographics
customer_id = np.arange(N)

age = np.clip(np.random.normal(42, 14, N), 18, 80).astype(int)

tenure_months = np.clip(np.random.exponential(36, N), 1, 120).astype(int)

region = np.random.choice(['US', 'EU', 'APAC', 'LATAM'], N, p=[0.40, 0.30, 0.20, 0.10])

gender = np.random.choice(['Male', 'Female', 'Non-binary'], N, p=[0.48, 0.48, 0.04])

plan_tier = np.random.choice(['Basic', 'Standard', 'Premium'], N, p=[0.30, 0.45, 0.25])

# Product & Engagement
num_products = np.clip(np.random.poisson(2.5, N), 1, 10).astype(int)

primary_product = np.random.choice(
    ['Photoshop', 'Premiere', 'Illustrator', 'InDesign', 'After Effects', 'Audition'],
    N, p=[0.30, 0.20, 0.20, 0.15, 0.10, 0.05]
)

days_since_last_login = np.clip(np.random.exponential(8, N), 0, 60).astype(int)

login_frequency_30d = np.clip(
    np.random.poisson(12, N) - days_since_last_login // 5, 0, 30
).astype(int)

feature_adoption_rate = np.random.beta(2, 4, N)

product_changes_6m = np.clip(np.random.poisson(0.8, N), 0, 5).astype(int)

mobile_app_user = (np.random.random(N) < 0.55).astype(int)

# Spending
monthly_spend_avg = np.clip(np.random.exponential(120, N) + 10, 10, 2000)

spend_trend = np.random.choice([-1, 0, 1], N, p=[0.20, 0.55, 0.25])

spend_volatility = np.clip(np.random.beta(1.5, 6, N) + 0.05, 0.05, 1.0)

monthly_spend_current = np.clip(
    monthly_spend_avg * (1 + spend_trend * 0.15 + np.random.normal(0, spend_volatility * 0.2, N)),
    10, 2000
)

num_txns_6m = np.clip(np.random.poisson(30, N), 1, 180).astype(int)

# Service & Sentiment
complaint_count_6m = np.clip(np.random.poisson(0.6, N), 0, 5).astype(int)

support_interactions_6m = np.clip(np.random.poisson(1.8, N), 0, 10).astype(int)

escalated_support = (np.random.random(N) < 0.15).astype(int)

# NPS correlated with sentiment; range -100 to 100
nps_base = np.random.normal(25, 40, N)
nps_score = np.clip(
    nps_base - complaint_count_6m * 15 - escalated_support * 25,
    -100, 100
).astype(int)

# Sentiment inversely correlated with complaints and escalations
sentiment_base = np.random.normal(68, 15, N)
sentiment_score = np.clip(
    sentiment_base
    - complaint_count_6m * 8
    - escalated_support * 15
    + (feature_adoption_rate - 0.3) * 20,
    0, 100
)

sentiment_trend = np.random.choice([-1, 0, 1], N, p=[0.20, 0.50, 0.30])
flip_mask = np.random.random(N) < 0.3
sentiment_trend = np.where(flip_mask, sentiment_trend, np.clip(spend_trend + np.random.randint(-1, 2, N), -1, 1))

# Contract
contract_type = np.where(np.random.random(N) < 0.70, 'annual', 'monthly')

discount_applied = (np.random.random(N) < 0.20).astype(int)

auto_renewal_enabled = (np.random.random(N) < 0.85).astype(int)

# Referral source
payment_failures_6m = np.clip(np.random.poisson(0.3, N), 0, 5).astype(int)

referral_source = np.random.choice(
    ['Organic', 'Paid', 'Referral', 'Social'], N, p=[0.35, 0.30, 0.20, 0.15]
)

# Churn target
logit = (
    -1.4                                          # intercept → ~13.5% base rate
    - (sentiment_score - 50) * 0.04              # high sentiment reduces churn
    + (spend_trend == -1).astype(float) * 1.2    # declining spend increases churn
    - num_products * 0.18                         # more products = switching costs
    - (tenure_months / 120) * 1.0                # longer tenure = loyal
    + (days_since_last_login / 60) * 1.0         # inactive = risky
    + escalated_support * 1.8                     # escalations = high risk
    + complaint_count_6m * 0.35                   # complaints increase churn
    - feature_adoption_rate * 1.2                 # high adoption = engaged
    + (contract_type == 'monthly').astype(float) * 0.6  # monthly = easier to leave
    - auto_renewal_enabled * 0.5                  # renewal enabled = sticky
    + (sentiment_trend == -1).astype(float) * 0.5
    - (sentiment_trend == 1).astype(float) * 0.3
    - (plan_tier == 'Premium').astype(float) * 0.3  # premium = more invested
    - login_frequency_30d * 0.02                  # frequent logins = engaged
    + payment_failures_6m * 0.4                   # payment failures increase churn
    + np.random.normal(0, 0.4, N)                # noise
)

churn_prob = 1 / (1 + np.exp(-logit))
churn = (np.random.random(N) < churn_prob).astype(int)

print(f"Churn rate: {churn.mean():.3f} ({churn.sum()} churned)")

df = pd.DataFrame({
    # Demographics (6)
    'customer_id': customer_id,
    'age': age,
    'tenure_months': tenure_months,
    'region': region,
    'gender': gender,
    'plan_tier': plan_tier,
    # Product & Engagement (7)
    'num_products': num_products,
    'primary_product': primary_product,
    'days_since_last_login': days_since_last_login,
    'login_frequency_30d': login_frequency_30d,
    'feature_adoption_rate': feature_adoption_rate.round(4),
    'product_changes_6m': product_changes_6m,
    'mobile_app_user': mobile_app_user,
    # Spending (5)
    'monthly_spend_avg': monthly_spend_avg.round(2),
    'monthly_spend_current': monthly_spend_current.round(2),
    'spend_trend': spend_trend,
    'spend_volatility': spend_volatility.round(4),
    'num_txns_6m': num_txns_6m,
    # Service & Sentiment (6)
    'sentiment_score': sentiment_score.round(1),
    'sentiment_trend': sentiment_trend,
    'nps_score': nps_score,
    'complaint_count_6m': complaint_count_6m,
    'support_interactions_6m': support_interactions_6m,
    'escalated_support': escalated_support,
    # Contract (4)
    'contract_type': contract_type,
    'discount_applied': discount_applied,
    'auto_renewal_enabled': auto_renewal_enabled,
    'referral_source': referral_source,
        'payment_failures_6m': np.clip(np.random.poisson(0.3, N), 0, 5).astype(int),
    # Target (1)
    'churn': churn,
})

assert df.isnull().sum().sum() == 0, "Missing values detected!"
assert len(df) == N, f"Expected {N} rows, got {len(df)}"
assert df.shape[1] == 30, f"Expected 30 columns, got {df.shape[1]}"

print(f"\nColumn count: {df.shape[1]}")
print("Columns:", list(df.columns))

print("\nCorrelations with churn (Pearson r):")
numeric_cols = [
    'sentiment_score', 'nps_score', 'spend_trend', 'num_products', 'tenure_months',
    'escalated_support', 'complaint_count_6m', 'days_since_last_login',
    'login_frequency_30d', 'feature_adoption_rate', 'auto_renewal_enabled',
]
for col in numeric_cols:
    r, p = stats.pearsonr(df[col], df['churn'])
    print(f"  {col:30s}: r={r:+.3f}  p={p:.2e}")

output_path = 'synthetic_churn_50k.csv'
df.to_csv(output_path, index=False)
print(f"\nSaved {len(df):,} rows × {df.shape[1]} columns → {output_path}")

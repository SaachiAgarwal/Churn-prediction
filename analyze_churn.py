import pandas as pd
import numpy as np
from scipy import stats

df = pd.read_csv('synthetic_churn_50k.csv')

# Derived features
df['engagement_score'] = (
    df['login_frequency_30d'] / 30 * 0.4
    + df['feature_adoption_rate'] * 0.4
    + (1 - df['days_since_last_login'] / 60) * 0.2
).clip(0, 1).round(4)

churners  = df[df['churn'] == 1]
retained  = df[df['churn'] == 0]

SEP  = '=' * 70
SEP2 = '-' * 70

# ── SECTION 1 ──────────────────────────────────────────────────────────────
print(f'\n{SEP}')
print('SECTION 1: BASELINE STATISTICS')
print(SEP)

total      = len(df)
churn_n    = df['churn'].sum()
churn_rate = churn_n / total

print(f'  Total customers : {total:,}')
print(f'  Churned         : {churn_n:,}')
print(f'  Retained        : {total - churn_n:,}')
print(f'  Churn rate      : {churn_rate:.1%}')

print(f'\n  {SEP2}')
print('  WHY ACCURACY IS MISLEADING')
print(f'  {SEP2}')
naive_acc    = (total - churn_n) / total
naive_recall = 0.0
print(f'  Naive model (predict "no churn" for everyone):')
print(f'    Accuracy : {naive_acc:.1%}   ← looks great on paper')
print(f'    Recall   : {naive_recall:.1%}   ← catches ZERO actual churners')
print(f'    Precision: N/A   (never predicts churn)')
print(f'  → A model must beat {naive_acc:.1%} accuracy AND achieve meaningful recall.')
print(f'    Use F1-score, AUC-ROC, or precision-recall curves instead.')

# ── SECTION 2 ──────────────────────────────────────────────────────────────
print(f'\n{SEP}')
print('SECTION 2: SEGMENT ANALYSIS')
print(SEP)

print('\n  Churn rate by PRIMARY PRODUCT')
print(f'  {SEP2}')
prod = (df.groupby('primary_product')['churn']
          .agg(['mean', 'sum', 'count'])
          .rename(columns={'mean':'churn_rate','sum':'churned','count':'total'})
          .sort_values('churn_rate', ascending=False))
for name, row in prod.iterrows():
    bar = '█' * int(row.churn_rate * 100)
    print(f'  {name:<16} {row.churn_rate:>6.1%}  {bar}  ({int(row.churned):,}/{int(row.total):,})')

print(f'\n  Churn rate by REGION')
print(f'  {SEP2}')
reg = (df.groupby('region')['churn']
         .agg(['mean', 'sum', 'count'])
         .rename(columns={'mean':'churn_rate','sum':'churned','count':'total'})
         .sort_values('churn_rate', ascending=False))
for name, row in reg.iterrows():
    bar = '█' * int(row.churn_rate * 100)
    print(f'  {name:<8} {row.churn_rate:>6.1%}  {bar}  ({int(row.churned):,}/{int(row.total):,})')

print(f'\n  Churn rate by TENURE SEGMENT')
print(f'  {SEP2}')
bins   = [0, 6, 12, 24, 999]
labels = ['0–6 months', '6–12 months', '12–24 months', '24+ months']
df['tenure_segment'] = pd.cut(df['tenure_months'], bins=bins, labels=labels)
ten = (df.groupby('tenure_segment', observed=True)['churn']
         .agg(['mean', 'sum', 'count'])
         .rename(columns={'mean':'churn_rate','sum':'churned','count':'total'}))
for name, row in ten.iterrows():
    bar = '█' * int(row.churn_rate * 100)
    print(f'  {str(name):<14} {row.churn_rate:>6.1%}  {bar}  ({int(row.churned):,}/{int(row.total):,})')

# ── SECTION 3 ──────────────────────────────────────────────────────────────
print(f'\n{SEP}')
print('SECTION 3: FEATURE CORRELATIONS WITH CHURN (Pearson r)')
print(SEP)

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols = [c for c in numeric_cols if c not in ('churn', 'customer_id')]

corrs = {}
for col in numeric_cols:
    r, _ = stats.pearsonr(df[col], df['churn'])
    corrs[col] = r

sorted_corrs = sorted(corrs.items(), key=lambda x: abs(x[1]), reverse=True)[:15]

print(f'\n  {"Rank":<5} {"Feature":<30} {"r":>8}  {"Direction"}')
print(f'  {SEP2}')
for i, (feat, r) in enumerate(sorted_corrs, 1):
    sign   = '+' if r >= 0 else ''
    arrow  = '↑ churn' if r > 0 else '↓ churn'
    print(f'  {i:<5} {feat:<30} {sign}{r:.4f}   {arrow}')

# ── SECTION 4 ──────────────────────────────────────────────────────────────
print(f'\n{SEP}')
print('SECTION 4: STATISTICAL TESTS — CHURNERS vs RETAINED (Welch t-test)')
print(SEP)

def sig_stars(p):
    if p < 0.001: return '***'
    if p < 0.01:  return '** '
    if p < 0.05:  return '*  '
    return '   '

test_features = [
    'sentiment_score',
    'monthly_spend_avg',
    'engagement_score',
    'days_since_last_login',
    'complaint_count_6m',
    'tenure_months',
    'num_products',
]

print(f'\n  {"Feature":<26} {"Churner μ":>10} {"Retained μ":>11} {"t-stat":>9} {"p-value":>12}  Sig')
print(f'  {SEP2}')
for feat in test_features:
    c_vals = churners[feat].dropna()
    r_vals = retained[feat].dropna()
    t, p   = stats.ttest_ind(c_vals, r_vals, equal_var=False)
    stars  = sig_stars(p)
    p_str  = f'{p:.2e}' if p < 0.0001 else f'{p:.4f}'
    print(f'  {feat:<26} {c_vals.mean():>10.3f} {r_vals.mean():>11.3f} {t:>9.3f} {p_str:>12}  {stars}')

print(f'\n  Significance: *** p<0.001  ** p<0.01  * p<0.05')

# ── SECTION 5 ──────────────────────────────────────────────────────────────
print(f'\n{SEP}')
print('SECTION 5: BEHAVIORAL PATTERNS')
print(SEP)

metrics = {
    'Sentiment score'       : 'sentiment_score',
    'Engagement score'      : 'engagement_score',
    'Monthly spend (avg)'   : 'monthly_spend_avg',
    'Num products'          : 'num_products',
}

print(f'\n  {"Metric":<26} {"Churners":>10} {"Retained":>10} {"Δ":>10}')
print(f'  {SEP2}')
for label, col in metrics.items():
    c_mean = churners[col].mean()
    r_mean = retained[col].mean()
    delta  = c_mean - r_mean
    sign   = '+' if delta >= 0 else ''
    print(f'  {label:<26} {c_mean:>10.3f} {r_mean:>10.3f} {sign}{delta:>9.3f}')

print(f'\n  Support Escalation Impact')
print(f'  {SEP2}')
esc_yes = df[df['escalated_support'] == 1]['churn'].mean()
esc_no  = df[df['escalated_support'] == 0]['churn'].mean()
lift    = esc_yes / esc_no
print(f'  Escalated support = Yes : {esc_yes:.1%} churn rate')
print(f'  Escalated support = No  : {esc_no:.1%} churn rate')
print(f'  Churn lift from escalation: {lift:.2f}x higher')

# ── SECTION 6 ──────────────────────────────────────────────────────────────
print(f'\n{SEP}')
print('SECTION 6: EARLY WARNING SIGNALS')
print(SEP)

print(f'\n  Churn rate by SENTIMENT TREND')
print(f'  {SEP2}')
trend_map = {-1: 'Declining', 0: 'Stable', 1: 'Improving'}
for val, label in trend_map.items():
    rate = df[df['sentiment_trend'] == val]['churn'].mean()
    bar  = '█' * int(rate * 100)
    print(f'  {label:<12} {rate:>6.1%}  {bar}')

print(f'\n  Churn rate by SPEND TREND')
print(f'  {SEP2}')
spend_map = {-1: 'Declining', 0: 'Stable', 1: 'Growing'}
for val, label in spend_map.items():
    rate = df[df['spend_trend'] == val]['churn'].mean()
    bar  = '█' * int(rate * 100)
    print(f'  {label:<12} {rate:>6.1%}  {bar}')

print(f'\n  Churn rate by DAYS SINCE LAST LOGIN')
print(f'  {SEP2}')
active   = df[df['days_since_last_login'] <= 30]['churn'].mean()
inactive = df[df['days_since_last_login'] >  30]['churn'].mean()
print(f'  Active   (≤30 days) : {active:.1%}')
print(f'  Inactive (>30 days) : {inactive:.1%}')
print(f'  Inactive customers churn {inactive/active:.2f}x more than active ones')

# ── SECTION 7 ──────────────────────────────────────────────────────────────
print(f'\n{SEP}')
print('SECTION 7: AT-RISK SEGMENT')
print(SEP)

at_risk = (
    (df['sentiment_score']       < 50) |
    (df['days_since_last_login'] > 30) |
    (df['complaint_count_6m']   >= 2)  |
    (df['spend_trend']          == -1)
)

df['at_risk'] = at_risk.astype(int)
n_at_risk      = at_risk.sum()
n_healthy      = (~at_risk).sum()
rate_at_risk   = df.loc[at_risk,  'churn'].mean()
rate_healthy   = df.loc[~at_risk, 'churn'].mean()
churners_caught= df.loc[at_risk & (df['churn'] == 1), 'churn'].sum()
coverage       = churners_caught / churn_n

print(f'\n  At-risk criteria:')
print(f'    • sentiment_score < 50')
print(f'    • days_since_last_login > 30')
print(f'    • complaint_count_6m ≥ 2')
print(f'    • spend_trend == -1 (declining)')

print(f'\n  {"Segment":<20} {"Customers":>12} {"% of Total":>12} {"Churn Rate":>12}')
print(f'  {SEP2}')
print(f'  {"At-risk":<20} {n_at_risk:>12,} {n_at_risk/total:>12.1%} {rate_at_risk:>12.1%}')
print(f'  {"Healthy":<20} {n_healthy:>12,} {n_healthy/total:>12.1%} {rate_healthy:>12.1%}')
print(f'  {"Total":<20} {total:>12,} {"100.0%":>12} {churn_rate:>12.1%}')

print(f'\n  Churn lift in at-risk segment: {rate_at_risk/rate_healthy:.2f}x vs healthy')
print(f'  Coverage: {coverage:.1%} of actual churners fall in the at-risk segment')
print(f'  Missed  : {1-coverage:.1%} of churners are in the "healthy" segment (silent churners)')

print(f'\n{SEP}')
print('END OF ANALYSIS')
print(SEP)

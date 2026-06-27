import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

SEP  = '=' * 70
SEP2 = '-' * 70

# ── Load ───────────────────────────────────────────────────────────────────
df_raw = pd.read_csv('synthetic_churn_50k.csv')
df     = df_raw.copy()
target = df['churn'].copy()

print(SEP)
print('FEATURE ENGINEERING PIPELINE')
print(SEP)
print(f'  Input  : {df.shape[0]:,} rows × {df.shape[1]} columns')

# Fill any unexpected missing values with median
if df.isnull().sum().sum() > 0:
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col].fillna(df[col].median(), inplace=True)
    print(f'  Missing: filled with column median')
else:
    print(f'  Missing: none found ✓')

feat = pd.DataFrame(index=df.index)   # accumulate all features here
feat['churn'] = target

# ══════════════════════════════════════════════════════════════════════════
# A. SPENDING PATTERNS
# ══════════════════════════════════════════════════════════════════════════
feat['monthly_spend_avg']      = df['monthly_spend_avg']
feat['monthly_spend_current']  = df['monthly_spend_current']

# One-hot: spend_trend  (-1 / 0 / 1)
feat['spend_declining']        = (df['spend_trend'] == -1).astype(int)
feat['spend_stable']           = (df['spend_trend'] ==  0).astype(int)
feat['spend_growing']          = (df['spend_trend'] ==  1).astype(int)

feat['monthly_spend_pct_change'] = (
    (df['monthly_spend_current'] - df['monthly_spend_avg'])
    / df['monthly_spend_avg'].replace(0, np.nan)
).fillna(0).round(4)

feat['monthly_spend_z_score']  = stats.zscore(df['monthly_spend_avg']).round(4)
feat['spend_volatility']       = df['spend_volatility']
feat['revenue_potential']      = (df['monthly_spend_avg'] * df['num_products']).round(2)
feat['high_spender_flag']      = (
    df['monthly_spend_avg'] >= df['monthly_spend_avg'].quantile(0.75)
).astype(int)

# ══════════════════════════════════════════════════════════════════════════
# B. ENGAGEMENT PATTERNS
# ══════════════════════════════════════════════════════════════════════════
feat['days_since_last_login']  = df['days_since_last_login']
feat['feature_adoption_rate']  = df['feature_adoption_rate']
feat['product_diversity']      = df['product_diversity']

feat['engagement_score'] = (
    df['login_frequency_30d'] / 30 * 0.4
    + df['feature_adoption_rate'] * 0.4
    + (1 - df['days_since_last_login'] / 60) * 0.2
).clip(0, 1).round(4)

feat['engagement_decline']     = (df['days_since_last_login'] > 30).astype(int)
feat['active_user_flag']       = (df['days_since_last_login'] < 7).astype(int)
feat['feature_adoption_flag']  = (
    df['feature_adoption_rate'] > df['feature_adoption_rate'].median()
).astype(int)
feat['product_expansion']      = (df['product_changes_6m'] > 0).astype(int)

# ══════════════════════════════════════════════════════════════════════════
# C. SENTIMENT & SERVICE
# ══════════════════════════════════════════════════════════════════════════
feat['sentiment_score']        = df['sentiment_score']

# One-hot: sentiment_trend
feat['sentiment_declining']    = (df['sentiment_trend'] == -1).astype(int)
feat['sentiment_stable']       = (df['sentiment_trend'] ==  0).astype(int)
feat['sentiment_improving']    = (df['sentiment_trend'] ==  1).astype(int)

feat['sentiment_decline_flag'] = feat['sentiment_declining']   # alias for clarity
feat['complaint_count_6m']     = df['complaint_count_6m']

feat['complaint_density'] = (
    df['complaint_count_6m']
    / df['support_interactions_6m'].replace(0, np.nan)
).fillna(0).round(4)

feat['complaint_flag']         = (df['complaint_count_6m'] >= 2).astype(int)

feat['support_intensity'] = (
    df['support_interactions_6m']
    / df['tenure_months'].replace(0, np.nan)
).fillna(0).round(4)

feat['escalation_flag']        = df['escalated_support']

# composite: low sentiment + has complaints = escalation risk score (0–1 normalized)
feat['escalation_risk'] = (
    (1 - df['sentiment_score'] / 100) * 0.6
    + (df['complaint_count_6m'] / 5) * 0.4
).clip(0, 1).round(4)

feat['sentiment_z_score']      = stats.zscore(df['sentiment_score']).round(4)

# ══════════════════════════════════════════════════════════════════════════
# D. LIFECYCLE & TENURE
# ══════════════════════════════════════════════════════════════════════════
feat['tenure_months']          = df['tenure_months']

# One-hot: tenure segments
feat['tenure_0_6m']            = (df['tenure_months'] <   6).astype(int)
feat['tenure_6_12m']           = (df['tenure_months'].between(6,  12, inclusive='left')).astype(int)
feat['tenure_12_24m']          = (df['tenure_months'].between(12, 24, inclusive='left')).astype(int)
feat['tenure_24m_plus']        = (df['tenure_months'] >= 24).astype(int)

feat['new_customer_flag']      = feat['tenure_0_6m']
feat['mature_customer_flag']   = feat['tenure_24m_plus']
feat['contract_type_numeric']  = (df['contract_type'] == 'annual').astype(int)
feat['contract_stability']     = df['auto_renewal_enabled']
feat['discount_applied']       = df['discount_applied']

feat['contract_risk'] = (
    (df['contract_type'] == 'monthly') & (df['auto_renewal_enabled'] == 0)
).astype(int)

# ══════════════════════════════════════════════════════════════════════════
# E. PAYMENT RELIABILITY
# ══════════════════════════════════════════════════════════════════════════
feat['failed_payments_count']  = df['payment_failures_6m']

feat['failed_payments_rate'] = (
    df['payment_failures_6m']
    / df['num_txns_6m'].replace(0, np.nan)
).fillna(0).round(4)

# payment_method_changes — not in raw data; proxy: product_changes as plan changes
feat['payment_method_changes'] = df['product_changes_6m']
feat['payment_risk_flag']      = (df['payment_failures_6m'] > 0).astype(int)

# ══════════════════════════════════════════════════════════════════════════
# F. PRODUCT PORTFOLIO
# ══════════════════════════════════════════════════════════════════════════
feat['num_products']           = df['num_products']

# One-hot: primary_product (6 categories)
for prod in df['primary_product'].unique():
    col = 'product_' + prod.lower().replace(' ', '_')
    feat[col] = (df['primary_product'] == prod).astype(int)

feat['product_concentration']  = (1 / df['num_products']).round(4)
feat['product_risk']           = (df['num_products'] <= 1).astype(int)

# ══════════════════════════════════════════════════════════════════════════
# BONUS: COMPOSITE RISK FEATURES
# ══════════════════════════════════════════════════════════════════════════
feat['churn_risk_score'] = (
    feat['sentiment_declining']   * 0.15
    + feat['escalation_flag']     * 0.25
    + feat['complaint_flag']      * 0.15
    + feat['spend_declining']     * 0.20
    + feat['engagement_decline']  * 0.10
    + feat['contract_risk']       * 0.10
    + feat['payment_risk_flag']   * 0.05
).round(4)

feat['loyalty_score'] = (
    feat['contract_type_numeric'] * 0.20
    + feat['contract_stability']  * 0.15
    + feat['mature_customer_flag']* 0.15
    + feat['active_user_flag']    * 0.20
    + feat['feature_adoption_flag']* 0.15
    + feat['high_spender_flag']   * 0.15
).round(4)

feat['num_txns_6m']            = df['num_txns_6m']
feat['login_frequency_30d']    = df['login_frequency_30d']
feat['support_interactions_6m']= df['support_interactions_6m']
feat['age']                    = df['age']

# ══════════════════════════════════════════════════════════════════════════
# SCALE CONTINUOUS FEATURES
# ══════════════════════════════════════════════════════════════════════════
skip_scaling = {'churn'}
binary_cols  = [c for c in feat.columns
                if feat[c].dropna().isin([0, 1]).all() and c not in skip_scaling]
continuous   = [c for c in feat.columns
                if c not in binary_cols and c not in skip_scaling]

scaler = StandardScaler()
feat_scaled = feat.copy()
feat_scaled[continuous] = scaler.fit_transform(feat[continuous]).round(4)

# ══════════════════════════════════════════════════════════════════════════
# COLLINEARITY CHECK  (drop if |corr| > 0.95 with an earlier feature)
# ══════════════════════════════════════════════════════════════════════════
corr_matrix  = feat_scaled.drop(columns='churn').corr().abs()
upper        = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop      = [col for col in upper.columns if any(upper[col] > 0.95)]

feat_final   = feat_scaled.drop(columns=to_drop)

# ══════════════════════════════════════════════════════════════════════════
# VALIDATION REPORT
# ══════════════════════════════════════════════════════════════════════════
n_raw       = df_raw.shape[1] - 1          # exclude churn
n_engineered= feat_final.shape[1] - 1      # exclude churn
churn_col   = feat_final['churn']
feature_cols= [c for c in feat_final.columns if c != 'churn']

print(f'\n{SEP}')
print('PIPELINE SUMMARY')
print(SEP)
print(f'  Raw features           : {n_raw}')
print(f'  Engineered features    : {feat.shape[1] - 1}')
print(f'  Dropped (collinear)    : {len(to_drop)}  → {to_drop}')
print(f'  Final feature count    : {n_engineered}')
print(f'  Output shape           : {feat_final.shape[0]:,} rows × {feat_final.shape[1]} columns')

# ══════════════════════════════════════════════════════════════════════════
# FEATURE IMPORTANCE (correlation with churn)
# ══════════════════════════════════════════════════════════════════════════
corrs = {}
for col in feature_cols:
    r, _ = stats.pearsonr(feat_final[col], churn_col)
    corrs[col] = r

sorted_corrs = sorted(corrs.items(), key=lambda x: abs(x[1]), reverse=True)

print(f'\n{SEP}')
print('TOP 15 FEATURES BY CORRELATION WITH CHURN')
print(SEP)
print(f'  {"Rank":<5} {"Feature":<35} {"Pearson r":>10}  Direction')
print(f'  {SEP2}')
for i, (feat_name, r) in enumerate(sorted_corrs[:15], 1):
    sign  = '+' if r >= 0 else ''
    arrow = '↑ churn risk' if r > 0 else '↓ churn risk'
    print(f'  {i:<5} {feat_name:<35} {sign}{r:.4f}   {arrow}')

# ══════════════════════════════════════════════════════════════════════════
# FEATURE STATISTICS
# ══════════════════════════════════════════════════════════════════════════
print(f'\n{SEP}')
print('FEATURE STATISTICS (scaled values)')
print(SEP)
print(f'  {"Feature":<35} {"Mean":>8} {"Std":>8} {"Min":>8} {"Max":>8}')
print(f'  {SEP2}')
stats_df = feat_final[feature_cols].describe().T[['mean', 'std', 'min', 'max']]
for col, row in stats_df.iterrows():
    print(f'  {col:<35} {row["mean"]:>8.3f} {row["std"]:>8.3f} '
          f'{row["min"]:>8.3f} {row["max"]:>8.3f}')

# ══════════════════════════════════════════════════════════════════════════
# FEATURE CATEGORY BREAKDOWN
# ══════════════════════════════════════════════════════════════════════════
categories = {
    'A. Spending Patterns':    [c for c in feature_cols if any(k in c for k in
                                ['spend','revenue','high_spender'])],
    'B. Engagement Patterns':  [c for c in feature_cols if any(k in c for k in
                                ['login','adoption','engagement','active_user',
                                 'product_expansion','days_since'])],
    'C. Sentiment & Service':  [c for c in feature_cols if any(k in c for k in
                                ['sentiment','complaint','support','escalat'])],
    'D. Lifecycle & Tenure':   [c for c in feature_cols if any(k in c for k in
                                ['tenure','customer_flag','contract','discount'])],
    'E. Payment Reliability':  [c for c in feature_cols if any(k in c for k in
                                ['payment','failed'])],
    'F. Product Portfolio':    [c for c in feature_cols if any(k in c for k in
                                ['num_products','product_','concentration','product_risk'])],
    'G. Composite / Other':    [c for c in feature_cols if any(k in c for k in
                                ['risk_score','loyalty','txns','age'])],
}

print(f'\n{SEP}')
print('FEATURES BY CATEGORY')
print(SEP)
accounted = set()
for cat, cols in categories.items():
    unique_cols = [c for c in cols if c not in accounted]
    accounted.update(unique_cols)
    print(f'\n  {cat} ({len(unique_cols)} features)')
    for c in unique_cols:
        r = corrs.get(c, 0)
        sign = '+' if r >= 0 else ''
        print(f'    • {c:<38} r={sign}{r:.3f}')

remaining = [c for c in feature_cols if c not in accounted]
if remaining:
    print(f'\n  Uncategorised ({len(remaining)} features)')
    for c in remaining:
        r = corrs.get(c, 0)
        print(f'    • {c}  r={r:+.3f}')

# ══════════════════════════════════════════════════════════════════════════
# SAVE
# ══════════════════════════════════════════════════════════════════════════
feat_final.to_csv('synthetic_churn_features.csv', index=False)
print(f'\n{SEP}')
print('OUTPUT')
print(SEP)
print(f'  Saved: synthetic_churn_features.csv')
print(f'  Shape: {feat_final.shape[0]:,} rows × {feat_final.shape[1]} columns '
      f'({n_engineered} features + churn target)')
print(f'  Ready for ML training ✓')
print(SEP)

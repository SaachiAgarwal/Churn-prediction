import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ── Data ───────────────────────────────────────────────────────────────────
df = pd.read_csv('synthetic_churn_50k.csv')

df['engagement_score'] = (
    df['login_frequency_30d'] / 30 * 0.4
    + df['feature_adoption_rate'] * 0.4
    + (1 - df['days_since_last_login'] / 60) * 0.2
).clip(0, 1)

churn_n   = int(df['churn'].sum())
retain_n  = len(df) - churn_n
churn_pct = churn_n / len(df) * 100

GREEN = '#2CA02C'
RED   = '#D62728'

sns.set_style('whitegrid')
plt.rcParams.update({
    'font.family':      'DejaVu Sans',
    'axes.titlesize':   14,
    'axes.titleweight': 'bold',
    'axes.labelsize':   11,
    'xtick.labelsize':  10,
    'ytick.labelsize':  10,
    'figure.facecolor': 'white',
    'axes.facecolor':   'white',
})

# ── VIZ 1: Class Distribution ──────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Severe Class Imbalance: 13.5% Churn',
             fontsize=16, fontweight='bold', y=1.02)

# Pie
wedges, texts, autotexts = ax1.pie(
    [retain_n, churn_n],
    labels=['Retained', 'Churned'],
    colors=[GREEN, RED],
    autopct='%1.1f%%',
    startangle=90,
    wedgeprops=dict(edgecolor='white', linewidth=2.5),
    textprops=dict(fontsize=12),
    pctdistance=0.75,
)
for at in autotexts:
    at.set_fontsize(13)
    at.set_fontweight('bold')
    at.set_color('white')
ax1.set_title('Churn Proportion', pad=12)

# Bar
bars = ax2.bar(['Retained', 'Churned'], [retain_n, churn_n],
               color=[GREEN, RED], edgecolor='white', linewidth=1.5, width=0.45)
for bar, val in zip(bars, [retain_n, churn_n]):
    ax2.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 400,
             f'{val:,}', ha='center', fontsize=12, fontweight='bold')
ax2.set_title('Customer Count', pad=12)
ax2.set_ylabel('Number of Customers')
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x):,}'))
ax2.set_ylim(0, retain_n * 1.13)
sns.despine(ax=ax2, left=False, bottom=False)

plt.tight_layout()
plt.savefig('outputs/viz1_class_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print('Saved: viz1_class_distribution.png')

# ── VIZ 2: Churn by Product ────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))

prod = (df.groupby('primary_product')['churn']
          .mean()
          .sort_values(ascending=True)  # ascending so highest is at top of barh
          .reset_index())
prod.columns = ['product', 'churn_rate']
labels = prod['product'].tolist()
rates  = (prod['churn_rate'] * 100).tolist()
overall = churn_pct

bar_colors = [RED if r == max(rates) else '#F4A582' for r in rates]
bars = ax.barh(labels, rates, color=bar_colors, edgecolor='white',
               linewidth=1.2, height=0.55)

for bar, r in zip(bars, rates):
    ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
            f'{r:.1f}%', va='center', fontsize=11, fontweight='bold')

ax.axvline(overall, color='#333333', linestyle='--', linewidth=1.8,
           label=f'Overall avg ({overall:.1f}%)')
ax.set_xlabel('Churn Rate (%)', fontsize=12)
ax.set_title('Churn Varies Significantly by Product', pad=14)
ax.set_xlim(0, max(rates) + 3.5)
ax.legend(fontsize=10, framealpha=0.85)
sns.despine(ax=ax)

plt.tight_layout()
plt.savefig('outputs/viz2_churn_by_product.png', dpi=300, bbox_inches='tight')
plt.close()
print('Saved: viz2_churn_by_product.png')

# ── VIZ 3: Feature Correlations ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 7))

numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                if c not in ('churn', 'customer_id')]
corrs = {c: stats.pearsonr(df[c], df['churn'])[0] for c in numeric_cols}
top10 = sorted(corrs.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
feats = [f for f, _ in top10][::-1]
vals  = [v for _, v in top10][::-1]

colors = [RED if v > 0 else GREEN for v in vals]
bars   = ax.barh(feats, vals, color=colors, edgecolor='white',
                 linewidth=1.2, height=0.6)

for bar, v in zip(bars, vals):
    xpos = v + 0.004 if v >= 0 else v - 0.004
    ha   = 'left' if v >= 0 else 'right'
    ax.text(xpos, bar.get_y() + bar.get_height() / 2,
            f'{v:+.3f}', va='center', ha=ha, fontsize=10, fontweight='bold')

ax.axvline(0, color='#333333', linewidth=0.9)
ax.set_xlabel('Pearson Correlation with Churn', fontsize=12)
ax.set_title('Top 10 Features Predicting Churn', pad=14)

pos_patch = mpatches.Patch(color=RED,   label='Positive → increases churn')
neg_patch = mpatches.Patch(color=GREEN, label='Negative → decreases churn')
ax.legend(handles=[pos_patch, neg_patch], fontsize=10, framealpha=0.85)
sns.despine(ax=ax)

plt.tight_layout()
plt.savefig('outputs/viz3_feature_correlations.png', dpi=300, bbox_inches='tight')
plt.close()
print('Saved: viz3_feature_correlations.png')

# ── VIZ 4: Sentiment vs Churn (scatter with jitter) ───────────────────────
fig, ax = plt.subplots(figsize=(11, 6))

# Sample 4000 points so scatter isn't too dense
sample = df.sample(4000, random_state=42)
jitter = np.random.default_rng(42).uniform(-0.06, 0.06, len(sample))
colors_s = [RED if c == 1 else '#2E86AB' for c in sample['churn']]
alphas   = [0.55 if c == 1 else 0.20 for c in sample['churn']]

ax.scatter(sample['sentiment_score'], sample['churn'] + jitter,
           c=colors_s, alpha=0.35, s=18, linewidths=0)

# Overlay mean churn rate per sentiment bin
bins = pd.cut(df['sentiment_score'], bins=20)
bin_means = df.groupby(bins, observed=True)['churn'].mean()
bin_centers = [interval.mid for interval in bin_means.index]
ax.plot(bin_centers, bin_means.values, color='#333333',
        linewidth=2.5, label='Avg churn rate per bin', zorder=5)

ax.set_xlabel('Sentiment Score (0–100)', fontsize=12)
ax.set_ylabel('Churn (0 = Retained, 1 = Churned)', fontsize=12)
ax.set_title('Sentiment is Strongest Churn Predictor', pad=14)
ax.set_yticks([0, 1])
ax.set_yticklabels(['Retained (0)', 'Churned (1)'])

blue_patch = mpatches.Patch(color='#2E86AB', label='Retained')
red_patch  = mpatches.Patch(color=RED,       label='Churned')
line_patch = plt.Line2D([0], [0], color='#333333', linewidth=2.5,
                        label='Avg churn rate per bin')
ax.legend(handles=[blue_patch, red_patch, line_patch], fontsize=10, framealpha=0.85)
sns.despine(ax=ax)

plt.tight_layout()
plt.savefig('outputs/viz4_sentiment_vs_churn.png', dpi=300, bbox_inches='tight')
plt.close()
print('Saved: viz4_sentiment_vs_churn.png')

# ── VIZ 5: Engagement Score vs Churn ──────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Low Engagement = Higher Churn Risk', fontsize=15, fontweight='bold', y=1.02)

# Left: scatter with jitter — engagement score vs churn
sample2 = df.sample(4000, random_state=99)
jitter2  = np.random.default_rng(99).uniform(-0.06, 0.06, len(sample2))
colors_e = [RED if c == 1 else '#2E86AB' for c in sample2['churn']]

axes[0].scatter(sample2['engagement_score'], sample2['churn'] + jitter2,
                c=colors_e, alpha=0.30, s=16, linewidths=0)

# Trend line: avg churn per engagement bin
bins_e = pd.cut(df['engagement_score'], bins=20)
bin_means_e   = df.groupby(bins_e, observed=True)['churn'].mean()
bin_centers_e = [iv.mid for iv in bin_means_e.index]
axes[0].plot(bin_centers_e, bin_means_e.values, color='#333333',
             linewidth=2.5, label='Avg churn rate per bin', zorder=5)

axes[0].set_xlabel('Engagement Score (0–1)', fontsize=12)
axes[0].set_ylabel('Churn (0 = Retained, 1 = Churned)', fontsize=12)
axes[0].set_title('Engagement Score vs Churn', fontsize=12, fontweight='bold')
axes[0].set_yticks([0, 1])
axes[0].set_yticklabels(['Retained (0)', 'Churned (1)'])

blue_p = mpatches.Patch(color='#2E86AB', label='Retained')
red_p  = mpatches.Patch(color=RED,       label='Churned')
line_p = plt.Line2D([0], [0], color='#333333', linewidth=2.5, label='Avg churn rate per bin')
axes[0].legend(handles=[blue_p, red_p, line_p], fontsize=9, framealpha=0.85)
sns.despine(ax=axes[0])

# Right: churn rate per engagement decile (bar chart)
df['eng_decile'] = pd.qcut(df['engagement_score'], q=10, labels=False) + 1
decile_churn = df.groupby('eng_decile')['churn'].mean() * 100
decile_colors = [RED if v == decile_churn.max() else
                 GREEN if v == decile_churn.min() else '#7EB8D4'
                 for v in decile_churn.values]

bars = axes[1].bar(decile_churn.index, decile_churn.values,
                   color=decile_colors, edgecolor='white', linewidth=1.2, width=0.7)
for bar, v in zip(bars, decile_churn.values):
    axes[1].text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.2,
                 f'{v:.1f}%', ha='center', fontsize=8.5, fontweight='bold')

axes[1].axhline(churn_pct, color='#333333', linestyle='--',
                linewidth=1.5, label=f'Overall avg ({churn_pct:.1f}%)')
axes[1].set_xlabel('Engagement Decile (1 = Least, 10 = Most Engaged)', fontsize=11)
axes[1].set_ylabel('Churn Rate (%)', fontsize=12)
axes[1].set_title('Churn Rate by Engagement Decile', fontsize=12, fontweight='bold')
axes[1].set_xticks(range(1, 11))
axes[1].set_ylim(0, decile_churn.max() * 1.2)
axes[1].legend(fontsize=9, framealpha=0.85)
sns.despine(ax=axes[1])

plt.tight_layout()
plt.savefig('outputs/viz5_engagement_vs_churn.png', dpi=300, bbox_inches='tight')
plt.close()
print('Saved: viz5_engagement_vs_churn.png')

print('\nAll 5 visualizations saved to outputs/')

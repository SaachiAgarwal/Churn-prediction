import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ── Setup ──────────────────────────────────────────────────────────────────
df = pd.read_csv('synthetic_churn_50k.csv')

df['engagement_score'] = (
    df['login_frequency_30d'] / 30 * 0.4
    + df['feature_adoption_rate'] * 0.4
    + (1 - df['days_since_last_login'] / 60) * 0.2
).clip(0, 1).round(4)

PALETTE   = ['#2E86AB', '#E84855']   # blue=retained, red=churned
ACCENT    = '#F5A623'
BG        = '#F8F9FA'
GRID_C    = '#E0E0E0'
TEXT_C    = '#2C2C2C'
FONT      = 'DejaVu Sans'

sns.set_theme(style='whitegrid', font=FONT)
plt.rcParams.update({
    'figure.facecolor': BG,
    'axes.facecolor':   BG,
    'axes.edgecolor':   GRID_C,
    'axes.labelcolor':  TEXT_C,
    'text.color':       TEXT_C,
    'xtick.color':      TEXT_C,
    'ytick.color':      TEXT_C,
    'grid.color':       GRID_C,
    'font.family':      FONT,
})

churn_n    = df['churn'].sum()
retain_n   = len(df) - churn_n
churn_rate = churn_n / len(df)

# ── 1. CLASS DISTRIBUTION ──────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.patch.set_facecolor(BG)
fig.suptitle('Customer Churn — Class Distribution', fontsize=16, fontweight='bold',
             color=TEXT_C, y=1.01)

# Pie
wedges, texts, autotexts = axes[0].pie(
    [retain_n, churn_n],
    labels=['Retained', 'Churned'],
    colors=PALETTE,
    autopct='%1.1f%%',
    startangle=90,
    wedgeprops=dict(edgecolor='white', linewidth=2),
    textprops=dict(color=TEXT_C, fontsize=12),
)
for at in autotexts:
    at.set_fontsize(13)
    at.set_fontweight('bold')
    at.set_color('white')
axes[0].set_title('Proportion', fontsize=13, fontweight='bold', pad=12)

# Bar
bars = axes[1].bar(['Retained', 'Churned'], [retain_n, churn_n],
                   color=PALETTE, edgecolor='white', linewidth=1.5, width=0.5)
for bar, n in zip(bars, [retain_n, churn_n]):
    axes[1].text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 300,
                 f'{n:,}', ha='center', va='bottom',
                 fontsize=12, fontweight='bold', color=TEXT_C)
axes[1].set_title('Count', fontsize=13, fontweight='bold', pad=12)
axes[1].set_ylabel('Number of Customers', fontsize=11)
axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x):,}'))
axes[1].set_ylim(0, retain_n * 1.12)
axes[1].grid(axis='x', visible=False)

plt.tight_layout()
plt.savefig('plot1_class_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: plot1_class_distribution.png')

# ── 2. CHURN BY PRODUCT ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor(BG)

prod = (df.groupby('primary_product')['churn']
          .mean()
          .sort_values()
          .reset_index())
prod.columns = ['product', 'churn_rate']
prod_labels = prod['product'].tolist()
prod_rates  = prod['churn_rate'].tolist()

colors = [PALETTE[1] if r == max(prod_rates) else '#7EB8D4' for r in prod_rates]

bars = ax.barh(prod_labels, [r * 100 for r in prod_rates],
               color=colors, edgecolor='white', linewidth=1.2, height=0.55)

for bar, rate in zip(bars, prod_rates):
    ax.text(bar.get_width() + 0.15, bar.get_y() + bar.get_height() / 2,
            f'{rate:.1%}', va='center', fontsize=11, fontweight='bold', color=TEXT_C)

ax.axvline(churn_rate * 100, color=ACCENT, linestyle='--', linewidth=1.8,
           label=f'Overall avg ({churn_rate:.1%})')
ax.set_xlabel('Churn Rate (%)', fontsize=12)
ax.set_title('Churn Rate by Primary Product', fontsize=15, fontweight='bold', pad=14)
ax.set_xlim(0, max(prod_rates) * 100 + 3)
ax.legend(fontsize=10, framealpha=0.8)
ax.grid(axis='y', visible=False)

plt.tight_layout()
plt.savefig('plot2_churn_by_product.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: plot2_churn_by_product.png')

# ── 3. CHURN RATE VS ENGAGEMENT (ROC-style) ────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor(BG)
fig.suptitle('Churn Rate vs Engagement Score', fontsize=15, fontweight='bold',
             color=TEXT_C, y=1.01)

# Left: binned churn rate by engagement decile
df['eng_decile'] = pd.qcut(df['engagement_score'], q=10, labels=False) + 1
eng_churn = df.groupby('eng_decile')['churn'].mean() * 100

axes[0].plot(eng_churn.index, eng_churn.values,
             color=PALETTE[1], linewidth=2.5, marker='o', markersize=7)
axes[0].fill_between(eng_churn.index, eng_churn.values, alpha=0.15, color=PALETTE[1])
axes[0].axhline(churn_rate * 100, color=ACCENT, linestyle='--', linewidth=1.5,
                label=f'Avg churn ({churn_rate:.1%})')
axes[0].set_xlabel('Engagement Decile (1=Low, 10=High)', fontsize=11)
axes[0].set_ylabel('Churn Rate (%)', fontsize=11)
axes[0].set_title('Churn Rate by Engagement Decile', fontsize=12, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].set_xticks(range(1, 11))

# Right: cumulative capture curve (ROC-like)
df_sorted = df.sort_values('engagement_score')
n = len(df_sorted)
cum_churners = df_sorted['churn'].cumsum().values / churn_n
cum_pop      = np.arange(1, n + 1) / n

axes[1].plot(cum_pop * 100, cum_churners * 100,
             color=PALETTE[1], linewidth=2.5, label='Engagement model')
axes[1].plot([0, 100], [0, 100], color='grey', linestyle='--',
             linewidth=1.5, label='Random baseline')
axes[1].fill_between(cum_pop * 100, cum_churners * 100, cum_pop * 100,
                     alpha=0.12, color=PALETTE[1])

# Gini coefficient
gini = 1 - 2 * np.trapezoid(cum_churners, cum_pop)
axes[1].set_xlabel('% of Customers Ranked by Engagement (Low → High)', fontsize=10)
axes[1].set_ylabel('% of Churners Captured', fontsize=11)
axes[1].set_title(f'Cumulative Churn Capture Curve  (Gini={gini:.2f})', fontsize=12,
                  fontweight='bold')
axes[1].legend(fontsize=10)
axes[1].set_xlim(0, 100)
axes[1].set_ylim(0, 100)

plt.tight_layout()
plt.savefig('plot3_engagement_vs_churn.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: plot3_engagement_vs_churn.png')

# ── 4. FEATURE IMPORTANCE (TOP 10 CORRELATIONS) ────────────────────────────
fig, ax = plt.subplots(figsize=(11, 7))
fig.patch.set_facecolor(BG)

numeric_cols = df.select_dtypes(include=[np.number]).columns
numeric_cols = [c for c in numeric_cols if c not in ('churn', 'customer_id')]

corrs = {c: stats.pearsonr(df[c], df['churn'])[0] for c in numeric_cols}
top10 = sorted(corrs.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
feats, vals = zip(*top10)
feats = list(feats)[::-1]
vals  = list(vals)[::-1]

colors = [PALETTE[1] if v > 0 else PALETTE[0] for v in vals]
bars   = ax.barh(feats, vals, color=colors, edgecolor='white',
                 linewidth=1.2, height=0.6)

for bar, v in zip(bars, vals):
    xpos = v + 0.003 if v >= 0 else v - 0.003
    ha   = 'left' if v >= 0 else 'right'
    ax.text(xpos, bar.get_y() + bar.get_height() / 2,
            f'{v:+.4f}', va='center', ha=ha,
            fontsize=10, fontweight='bold', color=TEXT_C)

ax.axvline(0, color=TEXT_C, linewidth=0.8)
ax.set_xlabel('Pearson Correlation with Churn', fontsize=12)
ax.set_title('Top 10 Features Correlated with Churn', fontsize=15,
             fontweight='bold', pad=14)

pos_patch = mpatches.Patch(color=PALETTE[1], label='Positive (↑ churn)')
neg_patch = mpatches.Patch(color=PALETTE[0], label='Negative (↓ churn)')
ax.legend(handles=[pos_patch, neg_patch], fontsize=10, framealpha=0.8)
ax.grid(axis='y', visible=False)

plt.tight_layout()
plt.savefig('plot4_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: plot4_feature_importance.png')

# ── 5. FEATURE CORRELATION HEATMAP ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 11))
fig.patch.set_facecolor(BG)

heatmap_cols = [
    'churn', 'sentiment_score', 'escalated_support', 'complaint_count_6m',
    'spend_trend', 'sentiment_trend', 'engagement_score', 'feature_adoption_rate',
    'product_diversity', 'num_products', 'tenure_months', 'auto_renewal_enabled',
    'days_since_last_login', 'login_frequency_30d', 'monthly_spend_avg',
]

corr_matrix = df[heatmap_cols].corr()

mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)

cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(
    corr_matrix,
    mask=mask,
    cmap=cmap,
    center=0,
    vmin=-1, vmax=1,
    annot=True,
    fmt='.2f',
    annot_kws={'size': 8},
    linewidths=0.5,
    linecolor='white',
    square=True,
    ax=ax,
    cbar_kws={'shrink': 0.75, 'label': 'Pearson r'},
)

ax.set_title('Feature Correlation Heatmap', fontsize=16, fontweight='bold', pad=16)
ax.tick_params(axis='x', rotation=45, labelsize=9)
ax.tick_params(axis='y', rotation=0,  labelsize=9)

plt.tight_layout()
plt.savefig('plot5_correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: plot5_correlation_heatmap.png')

print('\nAll 5 plots saved successfully.')

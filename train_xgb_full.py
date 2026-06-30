import pandas as pd
import numpy as np
import pickle
import time
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model    import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing   import StandardScaler
from sklearn.metrics         import (roc_auc_score, average_precision_score,
                                     precision_score, recall_score, f1_score,
                                     confusion_matrix, roc_curve, precision_recall_curve)
import xgboost as xgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')
plt.rcParams.update({'font.family': 'DejaVu Sans',
                     'figure.facecolor': 'white', 'axes.facecolor': 'white'})

SEP  = '=' * 70
SEP2 = '-' * 70

# ── Load data ──────────────────────────────────────────────────────────────
df = pd.read_csv('synthetic_churn_features.csv')
y  = df['churn']
X  = df.drop(columns='churn')

print(SEP)
print('XGBoost (ALL 61 FEATURES) vs Logistic Regression')
print(SEP)
print(f'  Dataset : {X.shape[0]:,} rows × {X.shape[1]} features')
print(f'  Churn   : {y.mean():.1%} ({y.sum():,} churned)')

# ── Same split as train_models.py (same random_state) ─────────────────────
X_tv, X_test, y_tv, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(
    X_tv, y_tv, test_size=0.15/0.85, random_state=42, stratify=y_tv)

scaler     = StandardScaler()
X_train_sc = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
X_test_sc  = pd.DataFrame(scaler.transform(X_test),      columns=X.columns)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scale_pos = (y_train == 0).sum() / (y_train == 1).sum()

print(f'\n  Train : {len(X_train):,} | Test : {len(X_test):,}')
print(f'  scale_pos_weight : {scale_pos:.2f}')

# ── Helper ─────────────────────────────────────────────────────────────────
def eval_model(name, model, X_te, y_te):
    probs   = model.predict_proba(X_te)[:, 1]
    auc_roc = roc_auc_score(y_te, probs)
    auc_pr  = average_precision_score(y_te, probs)

    p, r, thr = precision_recall_curve(y_te, probs)
    f1s     = 2 * p * r / (p + r + 1e-9)
    opt_thr = thr[np.argmax(f1s[:-1])]
    preds   = (probs >= opt_thr).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_te, preds).ravel()
    fpr_arr, tpr_arr, _ = roc_curve(y_te, probs)

    # cost-optimal threshold
    thresholds   = np.linspace(0.01, 0.99, 500)
    net_values   = []
    for t in thresholds:
        pred_t = (probs >= t).astype(int)
        tn_t, fp_t, fn_t, tp_t = confusion_matrix(y_te, pred_t).ravel()
        net_values.append(tp_t * 5000 + fp_t * (-500))
    cost_thr   = thresholds[np.argmax(net_values)]
    cost_preds = (probs >= cost_thr).astype(int)
    tn_c, fp_c, fn_c, tp_c = confusion_matrix(y_te, cost_preds).ravel()
    net_val = tp_c * 5000 + fp_c * (-500)

    return {
        'name': name, 'probs': probs,
        'auc_roc': auc_roc, 'auc_pr': auc_pr,
        'f1_opt': f1s[np.argmax(f1s[:-1])],
        'precision_opt': precision_score(y_te, preds),
        'recall_opt': recall_score(y_te, preds),
        'opt_thr': opt_thr,
        'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn),
        'cost_thr': cost_thr, 'net_value': net_val,
        'tp_c': int(tp_c), 'fp_c': int(fp_c), 'fn_c': int(fn_c),
        'fpr_arr': fpr_arr, 'tpr_arr': tpr_arr,
        'pr_p': p, 'pr_r': r,
    }

def print_result(r):
    print(f'\n  AUC-ROC       : {r["auc_roc"]:.4f}')
    print(f'  AUC-PR        : {r["auc_pr"]:.4f}')
    print(f'  F1 (F1-opt)   : {r["f1_opt"]:.4f}  @ threshold={r["opt_thr"]:.3f}')
    print(f'  Precision     : {r["precision_opt"]:.4f}')
    print(f'  Recall        : {r["recall_opt"]:.4f}')
    print(f'  Confusion (F1-optimal threshold):')
    print(f'    TP={r["tp"]:,}  FP={r["fp"]:,}  FN={r["fn"]:,}  TN={r["tn"]:,}')
    print(f'  Cost-optimal threshold : {r["cost_thr"]:.3f}')
    print(f'    TP={r["tp_c"]:,}  FP={r["fp_c"]:,}  Net value=${r["net_value"]:,.0f}')

# ═══════════════════════════════════════════════════════════════════════════
# MODEL A — Logistic Regression (reload from pkl for exact same model)
# ═══════════════════════════════════════════════════════════════════════════
print(f'\n{SEP}')
print('MODEL A — LOGISTIC REGRESSION (best from earlier run)')
print(SEP)

with open('best_model.pkl', 'rb') as f:
    payload = pickle.load(f)

lr_best = payload['model']
rA = eval_model('Logistic Regression', lr_best, X_test_sc, y_test)
print_result(rA)

# ═══════════════════════════════════════════════════════════════════════════
# MODEL B — XGBoost (ALL 61 features)
# ═══════════════════════════════════════════════════════════════════════════
print(f'\n{SEP}')
print('MODEL B — XGBOOST (ALL 61 FEATURES)')
print(SEP)

xgb_params = {
    'max_depth':        [3, 5, 7],
    'learning_rate':    [0.01, 0.1, 0.2],
    'n_estimators':     [200],
    'subsample':        [0.8],
    'colsample_bytree': [0.8],
}

xgb_full = GridSearchCV(
    xgb.XGBClassifier(eval_metric='auc', use_label_encoder=False,
                      scale_pos_weight=scale_pos, random_state=42, n_jobs=-1),
    param_grid=xgb_params, cv=cv, scoring='roc_auc', n_jobs=1, verbose=0
)

t0 = time.perf_counter()
xgb_full.fit(X_train, y_train)
train_time = time.perf_counter() - t0
print(f'  Best params : {xgb_full.best_params_}')
print(f'  CV AUC      : {xgb_full.best_score_:.4f}')
print(f'  Train time  : {train_time:.1f}s')

rB = eval_model('XGBoost (All 61)', xgb_full.best_estimator_, X_test, y_test)
print_result(rB)

# ═══════════════════════════════════════════════════════════════════════════
# COMPARISON
# ═══════════════════════════════════════════════════════════════════════════
print(f'\n{SEP}')
print('HEAD-TO-HEAD COMPARISON')
print(SEP)

print(f'\n  {"Metric":<25} {"LR":>12} {"XGB-Full":>12} {"Winner":>12}')
print(f'  {SEP2}')

metrics = [
    ('AUC-ROC',      'auc_roc'),
    ('AUC-PR',       'auc_pr'),
    ('F1 (optimal)', 'f1_opt'),
    ('Precision',    'precision_opt'),
    ('Recall',       'recall_opt'),
]

for label, key in metrics:
    va, vb = rA[key], rB[key]
    winner = 'LR' if va > vb else 'XGB-Full'
    diff   = abs(va - vb)
    print(f'  {label:<25} {va:>12.4f} {vb:>12.4f} {winner:>12}  (Δ{diff:.4f})')

print(f'\n  {"Cost-optimal threshold":<25} {rA["cost_thr"]:>12.3f} {rB["cost_thr"]:>12.3f}')
print(f'  {"Net value ($)":<25} {rA["net_value"]:>12,.0f} {rB["net_value"]:>12,.0f}')
print(f'  {"TP at cost-optimal":<25} {rA["tp_c"]:>12,} {rB["tp_c"]:>12,}')
print(f'  {"FP at cost-optimal":<25} {rA["fp_c"]:>12,} {rB["fp_c"]:>12,}')

winner_overall = 'Logistic Regression' if rA['auc_roc'] > rB['auc_roc'] else 'XGBoost (All 61)'
print(f'\n  Overall winner by AUC-ROC: {winner_overall}')

# ═══════════════════════════════════════════════════════════════════════════
# FEATURE IMPORTANCE — XGBoost
# ═══════════════════════════════════════════════════════════════════════════
xgb_imp = pd.Series(xgb_full.best_estimator_.feature_importances_,
                    index=X.columns).sort_values(ascending=False)
lr_imp  = pd.Series(np.abs(lr_best.coef_[0]),
                    index=X.columns).sort_values(ascending=False)

print(f'\n  Top 10 features — XGBoost (All 61):')
for feat, val in xgb_imp.head(10).items():
    print(f'    {feat:<40} {val:.4f}')

print(f'\n  Top 10 features — Logistic Regression:')
for feat, val in lr_imp.head(10).items():
    print(f'    {feat:<40} {val:.4f}')

# ═══════════════════════════════════════════════════════════════════════════
# VISUALIZATIONS
# ═══════════════════════════════════════════════════════════════════════════

results = [rA, rB]
colors  = ['#2E86AB', '#D62728']

# ── Plot 1: ROC Curves ─────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 7))
for r, c in zip(results, colors):
    ax.plot(r['fpr_arr'], r['tpr_arr'], color=c, linewidth=2.5,
            label=f'{r["name"]}  (AUC={r["auc_roc"]:.4f})')
ax.plot([0,1],[0,1],'--',color='grey',linewidth=1.2,label='Random baseline')
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curves: LR vs XGBoost (All 61 Features)',
             fontsize=13, fontweight='bold', pad=12)
ax.legend(fontsize=11, loc='lower right', framealpha=0.9)
ax.set_xlim(0,1); ax.set_ylim(0,1)
sns.despine(ax=ax)
plt.tight_layout()
plt.savefig('outputs/xgb_full_roc.png', dpi=150, bbox_inches='tight')
plt.close()
print('\n  Saved: outputs/xgb_full_roc.png')

# ── Plot 2: Precision-Recall Curves ───────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 7))
for r, c in zip(results, colors):
    ax.plot(r['pr_r'], r['pr_p'], color=c, linewidth=2.5,
            label=f'{r["name"]}  (AUC-PR={r["auc_pr"]:.4f})')
ax.axhline(y_test.mean(), color='grey', linestyle='--', linewidth=1.2,
           label=f'No-skill baseline ({y_test.mean():.3f})')
ax.set_xlabel('Recall', fontsize=12)
ax.set_ylabel('Precision', fontsize=12)
ax.set_title('Precision-Recall: LR vs XGBoost (All 61 Features)',
             fontsize=13, fontweight='bold', pad=12)
ax.legend(fontsize=11, loc='upper right', framealpha=0.9)
ax.set_xlim(0,1); ax.set_ylim(0,1)
sns.despine(ax=ax)
plt.tight_layout()
plt.savefig('outputs/xgb_full_pr.png', dpi=150, bbox_inches='tight')
plt.close()
print('  Saved: outputs/xgb_full_pr.png')

# ── Plot 3: Metric Comparison Bar Chart ───────────────────────────────────
metric_labels = ['AUC-ROC', 'AUC-PR', 'F1', 'Precision', 'Recall']
metric_keys   = ['auc_roc', 'auc_pr', 'f1_opt', 'precision_opt', 'recall_opt']
x = np.arange(len(metric_labels))
width = 0.30

fig, ax = plt.subplots(figsize=(11, 6))
for i, (r, c) in enumerate(zip(results, colors)):
    vals = [r[k] for k in metric_keys]
    bars = ax.bar(x + i*width, vals, width, label=r['name'],
                  color=c, edgecolor='white', linewidth=1.2)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                f'{v:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.set_xticks(x + width/2)
ax.set_xticklabels(metric_labels, fontsize=12)
ax.set_ylabel('Score', fontsize=12)
ax.set_ylim(0, 1.1)
ax.set_title('LR vs XGBoost (All 61 Features) — Performance Metrics',
             fontsize=13, fontweight='bold', pad=12)
ax.legend(fontsize=11, framealpha=0.9)
ax.axhline(0.5, color='grey', linestyle=':', linewidth=1, alpha=0.5)
sns.despine(ax=ax)
plt.tight_layout()
plt.savefig('outputs/xgb_full_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print('  Saved: outputs/xgb_full_comparison.png')

# ── Plot 4: Feature Importance Side by Side ───────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 8))
fig.suptitle('Top 15 Feature Importance: LR vs XGBoost (All 61 Features)',
             fontsize=14, fontweight='bold', y=1.01)

for ax, imp, name, color in zip(axes,
        [lr_imp, xgb_imp],
        ['Logistic Regression', 'XGBoost (All 61)'],
        colors):
    top15 = imp.head(15)[::-1]
    bars  = ax.barh(top15.index, top15.values, color=color,
                    edgecolor='white', linewidth=0.8, height=0.7)
    for bar, v in zip(bars, top15.values):
        ax.text(bar.get_width() + top15.values.max()*0.01,
                bar.get_y() + bar.get_height()/2,
                f'{v:.3f}', va='center', fontsize=8, fontweight='bold')
    ax.set_title(name, fontsize=12, fontweight='bold', pad=10)
    ax.set_xlabel('Importance', fontsize=10)
    ax.set_xlim(0, top15.values.max() * 1.25)
    sns.despine(ax=ax)

plt.tight_layout()
plt.savefig('outputs/xgb_full_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print('  Saved: outputs/xgb_full_feature_importance.png')

# ── Plot 5: Net Value at cost-optimal threshold ───────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
model_names = [r['name'] for r in results]
net_vals    = [r['net_value'] for r in results]
bars = ax.bar(model_names, net_vals, color=colors, edgecolor='white',
              linewidth=1.5, width=0.4)
for bar, v in zip(bars, net_vals):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+20000,
            f'${v:,.0f}', ha='center', fontsize=12, fontweight='bold')
ax.set_ylabel('Net Value ($)', fontsize=12)
ax.set_title('Business Value at Cost-Optimal Threshold\n(TP=$5,000 benefit | FP=$500 cost)',
             fontsize=13, fontweight='bold', pad=12)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${int(x):,}'))
ax.set_ylim(0, max(net_vals) * 1.2)
sns.despine(ax=ax)
plt.tight_layout()
plt.savefig('outputs/xgb_full_net_value.png', dpi=150, bbox_inches='tight')
plt.close()
print('  Saved: outputs/xgb_full_net_value.png')

# ═══════════════════════════════════════════════════════════════════════════
# SAVE NEW BEST MODEL if XGB wins
# ═══════════════════════════════════════════════════════════════════════════
if rB['auc_roc'] > rA['auc_roc']:
    new_payload = {
        'model':      xgb_full.best_estimator_,
        'model_name': 'XGBoost (All 61)',
        'scaler':     None,
        'features':   list(X.columns),
        'metrics':    {k: rB[k] for k in ['auc_roc','auc_pr','f1_opt','precision_opt','recall_opt']},
    }
    with open('best_model_xgb_full.pkl', 'wb') as f:
        pickle.dump(new_payload, f)
    print('\n  Saved: best_model_xgb_full.pkl  (XGBoost wins — new best)')
else:
    print('\n  LR remains best model. best_model.pkl unchanged.')

print(f'\n{SEP}')
print('DONE')
print(SEP)

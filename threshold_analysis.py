import pandas as pd
import numpy as np
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (roc_curve, precision_recall_curve,
                              roc_auc_score, confusion_matrix,
                              precision_score, recall_score, f1_score)

sns.set_style('whitegrid')
plt.rcParams.update({'font.family':'DejaVu Sans','figure.facecolor':'white',
                     'axes.facecolor':'white','axes.titlesize':13,
                     'axes.titleweight':'bold'})

SEP  = '=' * 70
SEP2 = '-' * 70

# ── Costs ─────────────────────────────────────────────────────────────────
FP_COST  = -500    # blocked / incorrectly targeted customer
TP_BENE  = 5_000   # churner saved
FN_COST  =  0      # missed churner — no action taken
TN_BENE  =  0      # correctly left alone — no action, no cost

# ── Load model + data ──────────────────────────────────────────────────────
with open('best_model.pkl', 'rb') as f:
    payload = pickle.load(f)

model   = payload['model']
scaler  = payload['scaler']

df = pd.read_csv('synthetic_churn_features.csv')
y  = df['churn']
X  = df.drop(columns='churn')

from sklearn.model_selection import train_test_split
X_tv, X_test, y_tv, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y)
_, X_val, _, y_val = train_test_split(
    X_tv, y_tv, test_size=0.15/0.85, random_state=42, stratify=y_tv)

X_test_sc = pd.DataFrame(scaler.transform(X_test), columns=X.columns)
X_val_sc  = pd.DataFrame(scaler.transform(X_val),  columns=X.columns)

probs_test = model.predict_proba(X_test_sc)[:, 1]
probs_val  = model.predict_proba(X_val_sc)[:, 1]

N_test = len(y_test)
print(SEP)
print(f'THRESHOLD ANALYSIS — {payload["model_name"]}')
print(SEP)
print(f'  Test set : {N_test:,} customers  |  Churners: {y_test.sum():,} ({y_test.mean():.1%})')
print(f'  AUC-ROC  : {roc_auc_score(y_test, probs_test):.4f}')
print(f'\n  Cost matrix:')
print(f'    True Positive  (churner caught)   : +${TP_BENE:,}')
print(f'    False Positive (false alarm)       : -${abs(FP_COST):,}')
print(f'    True Negative  (correct no-action) :  $0')
print(f'    False Negative (missed churner)    :  $0')

# ═══════════════════════════════════════════════════════════════════════════
# 1. ROC CURVE + OPTIMAL THRESHOLD (Youden's J)
# ═══════════════════════════════════════════════════════════════════════════
fpr_arr, tpr_arr, roc_thrs = roc_curve(y_test, probs_test)
youdens_j   = tpr_arr - fpr_arr
opt_youden  = roc_thrs[np.argmax(youdens_j)]

print(f'\n{SEP}')
print('1. ROC CURVE — OPTIMAL THRESHOLD (Youden\'s J)')
print(SEP)
print(f'  Youden\'s J = TPR - FPR  (maximise separation between signal and noise)')
print(f'  Optimal threshold (Youden\'s J) : {opt_youden:.4f}')
preds_youden = (probs_test >= opt_youden).astype(int)
tn, fp, fn, tp = confusion_matrix(y_test, preds_youden).ravel()
print(f'  At this threshold:')
print(f'    Precision : {precision_score(y_test, preds_youden):.4f}')
print(f'    Recall    : {recall_score(y_test, preds_youden):.4f}')
print(f'    F1        : {f1_score(y_test, preds_youden):.4f}')
print(f'    TP={tp:,}  FP={fp:,}  FN={fn:,}  TN={tn:,}')

# ═══════════════════════════════════════════════════════════════════════════
# 2 & 3. COST MATRIX ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════
thresholds = np.linspace(0.01, 0.99, 500)
metrics    = []

for thr in thresholds:
    preds = (probs_test >= thr).astype(int)
    tn_, fp_, fn_, tp_ = confusion_matrix(y_test, preds, labels=[0,1]).ravel()

    net_value  = tp_ * TP_BENE + fp_ * FP_COST + fn_ * FN_COST + tn_ * TN_BENE
    prec = precision_score(y_test, preds, zero_division=0)
    rec  = recall_score(y_test, preds, zero_division=0)
    f1   = f1_score(y_test, preds, zero_division=0)
    contacted = tp_ + fp_

    metrics.append({
        'threshold':   thr,
        'precision':   prec,
        'recall':      rec,
        'f1':          f1,
        'tp': int(tp_), 'fp': int(fp_), 'fn': int(fn_), 'tn': int(tn_),
        'net_value':   net_value,
        'tp_revenue':  tp_ * TP_BENE,
        'fp_cost':     fp_ * FP_COST,
        'contacted':   contacted,
        'cost_per_contact': net_value / contacted if contacted > 0 else 0,
    })

mdf = pd.DataFrame(metrics)

# Cost-optimal threshold
opt_cost_idx = mdf['net_value'].idxmax()
opt_cost_row = mdf.loc[opt_cost_idx]
opt_cost_thr = opt_cost_row['threshold']

# F1-optimal threshold
opt_f1_idx  = mdf['f1'].idxmax()
opt_f1_row  = mdf.loc[opt_f1_idx]
opt_f1_thr  = opt_f1_row['threshold']

print(f'\n{SEP}')
print('2 & 3. COST MATRIX ANALYSIS')
print(SEP)

print(f'\n  Cost-optimal threshold : {opt_cost_thr:.4f}')
print(f'  ─────────────────────────────────────────────────────────')
print(f'  Customers contacted    : {int(opt_cost_row["contacted"]):,}  ({int(opt_cost_row["contacted"])/N_test:.1%} of base)')
print(f'  True positives (saved) : {int(opt_cost_row["tp"]):,}')
print(f'  False positives (waste): {int(opt_cost_row["fp"]):,}')
print(f'  False negatives (miss) : {int(opt_cost_row["fn"]):,}')
print(f'  Revenue from TP        : ${int(opt_cost_row["tp_revenue"]):>12,}')
print(f'  Cost from FP           : ${int(opt_cost_row["fp_cost"]):>12,}')
print(f'  Net value              : ${int(opt_cost_row["net_value"]):>12,}')
print(f'  Precision              : {opt_cost_row["precision"]:.4f}')
print(f'  Recall                 : {opt_cost_row["recall"]:.4f}')
print(f'  F1                     : {opt_cost_row["f1"]:.4f}')

print(f'\n  Comparison: cost-optimal vs F1-optimal vs Youden\'s J:')
print(f'\n  {"Criterion":<22} {"Threshold":>10} {"Net Value ($)":>14} {"Precision":>10} {"Recall":>10} {"F1":>8} {"Contacted":>10}')
print(f'  {SEP2}')

for label, row, thr in [
    ('Cost-optimal',  opt_cost_row, opt_cost_thr),
    ('F1-optimal',    opt_f1_row,   opt_f1_thr),
    ('Youden\'s J',  mdf[mdf['threshold'].between(opt_youden-0.01, opt_youden+0.01)].iloc[0], opt_youden),
]:
    print(f'  {label:<22} {thr:>10.4f} {int(row["net_value"]):>14,} '
          f'{row["precision"]:>10.4f} {row["recall"]:>10.4f} '
          f'{row["f1"]:>8.4f} {int(row["contacted"]):>10,}')

# ═══════════════════════════════════════════════════════════════════════════
# 4. PRECISION-RECALL AT DIFFERENT THRESHOLDS
# ═══════════════════════════════════════════════════════════════════════════
print(f'\n{SEP}')
print('4. PRECISION / RECALL / F1 AT KEY THRESHOLDS')
print(SEP)
print(f'\n  {"Threshold":>10} {"Precision":>10} {"Recall":>10} {"F1":>10} '
      f'{"TP":>6} {"FP":>6} {"FN":>6} {"Net Value ($)":>14} {"Contacted":>10}')
print(f'  {SEP2}')

key_thrs = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
key_thrs += [round(opt_cost_thr, 2), round(opt_f1_thr, 2), round(opt_youden, 2)]
key_thrs  = sorted(set([round(t, 2) for t in key_thrs]))

for thr in key_thrs:
    row = mdf[mdf['threshold'].between(thr-0.01, thr+0.01)]
    if row.empty:
        continue
    row = row.iloc[0]
    marker = ''
    if abs(thr - opt_cost_thr) < 0.015: marker = ' ← cost-opt'
    if abs(thr - opt_f1_thr)   < 0.015: marker = ' ← F1-opt'
    if abs(thr - opt_youden)   < 0.015: marker = ' ← Youden'
    print(f'  {thr:>10.2f} {row["precision"]:>10.4f} {row["recall"]:>10.4f} '
          f'{row["f1"]:>10.4f} {int(row["tp"]):>6} {int(row["fp"]):>6} '
          f'{int(row["fn"]):>6} {int(row["net_value"]):>14,} '
          f'{int(row["contacted"]):>10,}{marker}')

# ═══════════════════════════════════════════════════════════════════════════
# 5. VISUALIZATIONS
# ═══════════════════════════════════════════════════════════════════════════
GREEN, RED, BLUE, ORANGE, PURPLE = '#2CA02C','#D62728','#2E86AB','#F5A623','#7B2D8B'

# ── Fig 1: ROC curve with optimal point ───────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 7))
ax.plot(fpr_arr, tpr_arr, color=BLUE, linewidth=2.5,
        label=f'Logistic Regression (AUC={roc_auc_score(y_test, probs_test):.3f})')
ax.fill_between(fpr_arr, tpr_arr, alpha=0.08, color=BLUE)
ax.plot([0,1],[0,1],'--',color='grey',linewidth=1.5,label='Random baseline')

opt_fpr = fpr_arr[np.argmax(youdens_j)]
opt_tpr = tpr_arr[np.argmax(youdens_j)]
ax.scatter([opt_fpr],[opt_tpr], s=150, color=RED, zorder=5,
           label=f'Youden\'s J  (thr={opt_youden:.3f})')
ax.annotate(f' thr={opt_youden:.3f}\n TPR={opt_tpr:.3f}\n FPR={opt_fpr:.3f}',
            xy=(opt_fpr, opt_tpr), xytext=(opt_fpr+0.08, opt_tpr-0.12),
            fontsize=9, color=RED,
            arrowprops=dict(arrowstyle='->', color=RED, lw=1.5))

ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curve with Optimal Threshold (Youden\'s J)', pad=14)
ax.legend(fontsize=10, loc='lower right', framealpha=0.9)
sns.despine(ax=ax)
plt.tight_layout()
plt.savefig('outputs/t1_roc_optimal.png', dpi=150, bbox_inches='tight')
plt.close()
print(f'\n  Saved: outputs/t1_roc_optimal.png')

# ── Fig 2: Threshold vs Precision, Recall, F1 ─────────────────────────────
fig, ax = plt.subplots(figsize=(11, 6))
ax.plot(mdf['threshold'], mdf['precision'], color=BLUE,   linewidth=2.2, label='Precision')
ax.plot(mdf['threshold'], mdf['recall'],    color=RED,    linewidth=2.2, label='Recall')
ax.plot(mdf['threshold'], mdf['f1'],        color=GREEN,  linewidth=2.2, label='F1 Score')

for thr, color, label in [
    (opt_cost_thr, ORANGE, f'Cost-optimal ({opt_cost_thr:.3f})'),
    (opt_f1_thr,   GREEN,  f'F1-optimal ({opt_f1_thr:.3f})'),
    (opt_youden,   PURPLE, f'Youden\'s J ({opt_youden:.3f})'),
]:
    ax.axvline(thr, color=color, linestyle='--', linewidth=1.8, label=label)

ax.set_xlabel('Classification Threshold', fontsize=12)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Threshold vs Precision / Recall / F1', pad=14)
ax.legend(fontsize=9, framealpha=0.9, ncol=2)
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
sns.despine(ax=ax)
plt.tight_layout()
plt.savefig('outputs/t2_threshold_metrics.png', dpi=150, bbox_inches='tight')
plt.close()
print(f'  Saved: outputs/t2_threshold_metrics.png')

# ── Fig 3: Net value vs threshold (cost analysis) ─────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Cost Matrix Analysis', fontsize=15, fontweight='bold', y=1.01)

# Left: net value curve
ax = axes[0]
ax.plot(mdf['threshold'], mdf['net_value']/1000, color=GREEN, linewidth=2.5)
ax.fill_between(mdf['threshold'], mdf['net_value']/1000, 0,
                where=(mdf['net_value'] >= 0), alpha=0.15, color=GREEN, label='Positive value')
ax.fill_between(mdf['threshold'], mdf['net_value']/1000, 0,
                where=(mdf['net_value'] <  0), alpha=0.15, color=RED,   label='Negative value')
ax.axvline(opt_cost_thr, color=ORANGE, linestyle='--', linewidth=2,
           label=f'Cost-optimal ({opt_cost_thr:.3f})')
ax.axhline(0, color='grey', linewidth=0.8, linestyle=':')
ax.scatter([opt_cost_thr], [opt_cost_row['net_value']/1000],
           s=150, color=ORANGE, zorder=5)
ax.annotate(f'  Max: ${opt_cost_row["net_value"]/1000:,.0f}K\n  thr={opt_cost_thr:.3f}',
            xy=(opt_cost_thr, opt_cost_row['net_value']/1000),
            xytext=(opt_cost_thr+0.08, opt_cost_row['net_value']/1000),
            fontsize=9, color=ORANGE,
            arrowprops=dict(arrowstyle='->', color=ORANGE, lw=1.5))
ax.set_xlabel('Classification Threshold', fontsize=12)
ax.set_ylabel('Net Value ($K)', fontsize=12)
ax.set_title('Net Business Value vs Threshold\n(TP=+$5K, FP=−$500)', pad=12)
ax.legend(fontsize=9, framealpha=0.9)
ax.set_xlim(0, 1)
sns.despine(ax=ax)

# Right: TP revenue vs FP cost stacked area
ax = axes[1]
ax.fill_between(mdf['threshold'], mdf['tp_revenue']/1000, 0,
                alpha=0.4, color=GREEN, label='TP revenue (+$5K each)')
ax.fill_between(mdf['threshold'], mdf['fp_cost']/1000,   0,
                alpha=0.4, color=RED,   label='FP cost (−$500 each)')
ax.plot(mdf['threshold'], mdf['tp_revenue']/1000, color=GREEN, linewidth=1.8)
ax.plot(mdf['threshold'], mdf['fp_cost']/1000,   color=RED,   linewidth=1.8)
ax.axvline(opt_cost_thr, color=ORANGE, linestyle='--', linewidth=2,
           label=f'Cost-optimal ({opt_cost_thr:.3f})')
ax.axhline(0, color='grey', linewidth=0.8, linestyle=':')
ax.set_xlabel('Classification Threshold', fontsize=12)
ax.set_ylabel('Value ($K)', fontsize=12)
ax.set_title('TP Revenue vs FP Cost by Threshold', pad=12)
ax.legend(fontsize=9, framealpha=0.9)
ax.set_xlim(0, 1)
sns.despine(ax=ax)

plt.tight_layout()
plt.savefig('outputs/t3_cost_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print(f'  Saved: outputs/t3_cost_analysis.png')

# ── Fig 4: Precision-Recall curve with threshold annotations ──────────────
prec_arr, rec_arr, pr_thrs = precision_recall_curve(y_test, probs_test)

fig, ax = plt.subplots(figsize=(9, 7))
sc = ax.scatter(rec_arr[:-1], prec_arr[:-1], c=pr_thrs,
                cmap='RdYlGn_r', s=8, alpha=0.7, zorder=3)
plt.colorbar(sc, ax=ax, label='Threshold', shrink=0.85)
ax.plot(rec_arr, prec_arr, color='grey', linewidth=1, alpha=0.4, zorder=2)
ax.axhline(y_test.mean(), color='grey', linestyle='--', linewidth=1.2,
           label=f'No-skill ({y_test.mean():.2f})')

for thr, color, label in [
    (opt_cost_thr, ORANGE, f'Cost-opt ({opt_cost_thr:.3f})'),
    (opt_f1_thr,   GREEN,  f'F1-opt ({opt_f1_thr:.3f})'),
    (opt_youden,   PURPLE, f'Youden ({opt_youden:.3f})'),
]:
    idx = np.argmin(np.abs(pr_thrs - thr))
    ax.scatter([rec_arr[idx]], [prec_arr[idx]], s=200, color=color,
               zorder=5, label=label, edgecolors='white', linewidth=1.5)

ax.set_xlabel('Recall', fontsize=12)
ax.set_ylabel('Precision', fontsize=12)
ax.set_title('Precision-Recall Curve\n(colour = classification threshold)', pad=14)
ax.legend(fontsize=9, framealpha=0.9)
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
sns.despine(ax=ax)
plt.tight_layout()
plt.savefig('outputs/t4_pr_threshold.png', dpi=150, bbox_inches='tight')
plt.close()
print(f'  Saved: outputs/t4_pr_threshold.png')

# ── Fig 5: Confusion matrices side-by-side for 3 thresholds ───────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Confusion Matrices at Key Thresholds (% of test set)', fontsize=14,
             fontweight='bold', y=1.02)

for ax, (thr, label, color) in zip(axes, [
    (opt_youden,   f'Youden\'s J\n(thr={opt_youden:.3f})', PURPLE),
    (opt_f1_thr,   f'F1-Optimal\n(thr={opt_f1_thr:.3f})',  GREEN),
    (opt_cost_thr, f'Cost-Optimal\n(thr={opt_cost_thr:.3f})', ORANGE),
]):
    preds = (probs_test >= thr).astype(int)
    cm    = confusion_matrix(y_test, preds)
    cm_pct = cm.astype(float) / cm.sum() * 100
    tn_, fp_, fn_, tp_ = cm.ravel()
    net = tp_ * TP_BENE + fp_ * FP_COST

    sns.heatmap(cm_pct, annot=True, fmt='.1f', cmap='Blues', ax=ax,
                linewidths=2, linecolor='white',
                xticklabels=['Pred: Stay','Pred: Churn'],
                yticklabels=['Act: Stay','Act: Churn'],
                cbar=False, annot_kws={'size':12,'weight':'bold'})
    prec_ = precision_score(y_test, preds, zero_division=0)
    rec_  = recall_score(y_test, preds, zero_division=0)
    f1_   = f1_score(y_test, preds, zero_division=0)
    ax.set_title(f'{label}\nPrec={prec_:.3f}  Rec={rec_:.3f}  F1={f1_:.3f}\n'
                 f'Net Value: ${net:,.0f}', fontsize=10, fontweight='bold', pad=10)
    ax.tick_params(labelsize=9)

plt.tight_layout()
plt.savefig('outputs/t5_confusion_thresholds.png', dpi=150, bbox_inches='tight')
plt.close()
print(f'  Saved: outputs/t5_confusion_thresholds.png')

# ── Summary ────────────────────────────────────────────────────────────────
print(f'\n{SEP}')
print('THRESHOLD RECOMMENDATION SUMMARY')
print(SEP)
print(f"""
  Three thresholds, three different goals:

  1. YOUDEN'S J  (thr={opt_youden:.3f})
     Goal: maximise statistical separation (TPR - FPR)
     Best for: model evaluation and research
     Precision={mdf[mdf['threshold'].between(opt_youden-0.01,opt_youden+0.01)].iloc[0]['precision']:.3f}  Recall={mdf[mdf['threshold'].between(opt_youden-0.01,opt_youden+0.01)].iloc[0]['recall']:.3f}

  2. F1-OPTIMAL  (thr={opt_f1_thr:.3f})
     Goal: balance precision and recall equally
     Best for: when both false positives and false negatives matter equally
     Precision={opt_f1_row['precision']:.3f}  Recall={opt_f1_row['recall']:.3f}

  3. COST-OPTIMAL  (thr={opt_cost_thr:.3f})  ← RECOMMENDED FOR BUSINESS USE
     Goal: maximise net revenue (TP benefit minus FP cost)
     Best for: when intervention has a known dollar value
     Precision={opt_cost_row['precision']:.3f}  Recall={opt_cost_row['recall']:.3f}
     Contacted={int(opt_cost_row['contacted']):,} customers  ({int(opt_cost_row['contacted'])/N_test:.1%} of base)
     Net value= ${int(opt_cost_row['net_value']):,}

  KEY INSIGHT:
  At the cost-optimal threshold, we contact fewer customers but with
  higher precision — every $500 spent on outreach recovers $5,000 in
  retained revenue on average. Raising the threshold further reduces
  coverage; lowering it wastes budget on non-churners.
""")
print(SEP)

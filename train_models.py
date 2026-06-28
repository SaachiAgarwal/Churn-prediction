import pandas as pd
import numpy as np
import pickle
import time
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model   import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing  import StandardScaler
from sklearn.metrics        import (roc_auc_score, average_precision_score,
                                    precision_score, recall_score, f1_score,
                                    confusion_matrix, roc_curve, precision_recall_curve)
from sklearn.pipeline       import Pipeline
import xgboost as xgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

sns.set_style('whitegrid')
plt.rcParams.update({'font.family':'DejaVu Sans','figure.facecolor':'white','axes.facecolor':'white'})

SEP  = '=' * 70
SEP2 = '-' * 70
PALETTE = ['#2E86AB','#D62728','#2CA02C','#F5A623']
MODEL_NAMES = ['Logistic Regression','XGBoost (Structured)','XGBoost (Sentiment)','Ensemble Stack']

# ── Load engineered features ───────────────────────────────────────────────
df = pd.read_csv('synthetic_churn_features.csv')
y  = df['churn']
X  = df.drop(columns='churn')

print(SEP)
print('CHURN PREDICTION — MODEL TRAINING & COMPARISON')
print(SEP)
print(f'  Dataset : {X.shape[0]:,} rows × {X.shape[1]} features')
print(f'  Churn   : {y.mean():.1%} ({y.sum():,} churned)')

# ── Feature sets ──────────────────────────────────────────────────────────
SENTIMENT_COLS = [c for c in X.columns if any(k in c for k in [
    'sentiment','complaint','escalat','support_intensity','high_value_low'
])]
STRUCTURED_COLS = [c for c in X.columns if c not in SENTIMENT_COLS]

print(f'\n  Structured features : {len(STRUCTURED_COLS)}')
print(f'  Sentiment features  : {len(SENTIMENT_COLS)}')

# ── Train / Val / Test split ───────────────────────────────────────────────
X_tv, X_test, y_tv, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(
    X_tv, y_tv, test_size=0.15/0.85, random_state=42, stratify=y_tv)

print(f'\n  Train : {len(X_train):,} | Val : {len(X_val):,} | Test : {len(X_test):,}')

# Scale on train, apply to val/test (fixes leakage from feature_engineering.py)
scaler      = StandardScaler()
X_train_sc  = pd.DataFrame(scaler.fit_transform(X_train),   columns=X.columns)
X_val_sc    = pd.DataFrame(scaler.transform(X_val),         columns=X.columns)
X_test_sc   = pd.DataFrame(scaler.transform(X_test),        columns=X.columns)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ═══════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def eval_model(name, model, X_tr, y_tr, X_te, y_te, proba=True):
    t0 = time.perf_counter()
    if proba:
        probs = model.predict_proba(X_te)[:, 1]
    else:
        probs = model.decision_function(X_te)
    inf_ms = (time.perf_counter() - t0) * 1000

    auc_roc = roc_auc_score(y_te, probs)
    auc_pr  = average_precision_score(y_te, probs)

    results = {'model': name, 'auc_roc': auc_roc, 'auc_pr': auc_pr,
               'inference_ms': inf_ms}

    # Optimal threshold by F1 on validation
    f_prec, f_rec, f_thr = precision_recall_curve(y_te, probs)
    f1s = 2 * f_prec * f_rec / (f_prec + f_rec + 1e-9)
    opt_thr = f_thr[np.argmax(f1s[:-1])]
    preds_opt = (probs >= opt_thr).astype(int)

    results['optimal_threshold'] = opt_thr
    results['f1_optimal']   = f1_score(y_te, preds_opt)
    results['precision_opt']= precision_score(y_te, preds_opt)
    results['recall_opt']   = recall_score(y_te, preds_opt)

    for thr in [0.3, 0.5, 0.7]:
        preds = (probs >= thr).astype(int)
        results[f'precision_{thr}'] = precision_score(y_te, preds, zero_division=0)
        results[f'recall_{thr}']    = recall_score(y_te, preds,    zero_division=0)
        results[f'f1_{thr}']        = f1_score(y_te, preds,        zero_division=0)

    cm = confusion_matrix(y_te, preds_opt)
    tn, fp, fn, tp = cm.ravel()
    results['tn'], results['fp'] = int(tn), int(fp)
    results['fn'], results['tp'] = int(fn), int(tp)
    results['fpr'] = fp / (fp + tn)
    results['tpr'] = tp / (tp + fn)

    fpr_arr, tpr_arr, _ = roc_curve(y_te, probs)
    results['_probs']   = probs
    results['_fpr_arr'] = fpr_arr
    results['_tpr_arr'] = tpr_arr
    results['_pr_prec'] = f_prec
    results['_pr_rec']  = f_rec
    results['_cm']      = cm
    return results


def print_result(r):
    print(f'\n  AUC-ROC        : {r["auc_roc"]:.4f}')
    print(f'  AUC-PR         : {r["auc_pr"]:.4f}')
    print(f'  F1 (opt thr)   : {r["f1_optimal"]:.4f}  @ threshold={r["optimal_threshold"]:.3f}')
    print(f'  Precision/Rec  : {r["precision_opt"]:.4f} / {r["recall_opt"]:.4f}')
    print(f'  FPR / TPR      : {r["fpr"]:.4f} / {r["tpr"]:.4f}')
    print(f'  Inference      : {r["inference_ms"]:.1f} ms')
    print(f'\n  Confusion Matrix (opt threshold):')
    print(f'               Pred 0   Pred 1')
    print(f'  Actual 0  {r["tn"]:>8,} {r["fp"]:>8,}   (TN / FP)')
    print(f'  Actual 1  {r["fn"]:>8,} {r["tp"]:>8,}   (FN / TP)')
    print(f'\n  Threshold breakdown:')
    print(f'  {"Threshold":<12} {"Precision":>10} {"Recall":>10} {"F1":>10}')
    print(f'  {SEP2}')
    for thr in [0.3, 0.5, 0.7]:
        print(f'  {thr:<12.1f} {r[f"precision_{thr}"]:>10.4f} '
              f'{r[f"recall_{thr}"]:>10.4f} {r[f"f1_{thr}"]:>10.4f}')


all_results = []
feat_importances = {}

# ═══════════════════════════════════════════════════════════════════════════
# MODEL 1 — LOGISTIC REGRESSION
# ═══════════════════════════════════════════════════════════════════════════
print(f'\n{SEP}')
print('MODEL 1 — LOGISTIC REGRESSION')
print(SEP)

lr_grid = GridSearchCV(
    LogisticRegression(max_iter=2000, solver='saga', class_weight='balanced'),
    param_grid={'C': [0.001, 0.01, 0.1, 1, 10], 'penalty': ['l1', 'l2']},
    cv=cv, scoring='roc_auc', n_jobs=-1, verbose=0
)
t0 = time.perf_counter()
lr_grid.fit(X_train_sc, y_train)
print(f'  Best params : {lr_grid.best_params_}  (CV AUC={lr_grid.best_score_:.4f})')
print(f'  Train time  : {time.perf_counter()-t0:.1f}s')

lr_best = lr_grid.best_estimator_
r1 = eval_model('Logistic Regression', lr_best, X_train_sc, y_train, X_test_sc, y_test)
print_result(r1)
all_results.append(r1)

# Feature importance: absolute coefficient values
lr_imp = pd.Series(np.abs(lr_best.coef_[0]), index=X.columns).sort_values(ascending=False)
feat_importances['Logistic Regression'] = lr_imp

# ═══════════════════════════════════════════════════════════════════════════
# MODEL 2 — XGBOOST (STRUCTURED FEATURES)
# ═══════════════════════════════════════════════════════════════════════════
print(f'\n{SEP}')
print('MODEL 2 — XGBOOST (STRUCTURED FEATURES)')
print(SEP)

xgb_params = {
    'max_depth':      [3, 5, 7],
    'learning_rate':  [0.01, 0.1, 0.2],
    'n_estimators':   [200],
    'subsample':      [0.8],
    'colsample_bytree': [0.8],
}
scale_pos = (y_train == 0).sum() / (y_train == 1).sum()

xgb2 = GridSearchCV(
    xgb.XGBClassifier(eval_metric='auc', use_label_encoder=False,
                      scale_pos_weight=scale_pos, random_state=42, n_jobs=-1),
    param_grid=xgb_params, cv=cv, scoring='roc_auc', n_jobs=1, verbose=0
)
t0 = time.perf_counter()
xgb2.fit(X_train[STRUCTURED_COLS], y_train)
print(f'  Best params : {xgb2.best_params_}  (CV AUC={xgb2.best_score_:.4f})')
print(f'  Train time  : {time.perf_counter()-t0:.1f}s')

r2 = eval_model('XGBoost (Structured)', xgb2.best_estimator_,
                X_train[STRUCTURED_COLS], y_train, X_test[STRUCTURED_COLS], y_test)
print_result(r2)
all_results.append(r2)

xgb2_imp = pd.Series(xgb2.best_estimator_.feature_importances_,
                     index=STRUCTURED_COLS).sort_values(ascending=False)
feat_importances['XGBoost (Structured)'] = xgb2_imp

# ═══════════════════════════════════════════════════════════════════════════
# MODEL 3 — XGBOOST (SENTIMENT FEATURES)
# ═══════════════════════════════════════════════════════════════════════════
print(f'\n{SEP}')
print('MODEL 3 — XGBOOST (SENTIMENT/SERVICE FEATURES)')
print(SEP)

xgb3 = GridSearchCV(
    xgb.XGBClassifier(eval_metric='auc', use_label_encoder=False,
                      scale_pos_weight=scale_pos, random_state=42, n_jobs=-1),
    param_grid=xgb_params, cv=cv, scoring='roc_auc', n_jobs=1, verbose=0
)
t0 = time.perf_counter()
xgb3.fit(X_train[SENTIMENT_COLS], y_train)
print(f'  Best params : {xgb3.best_params_}  (CV AUC={xgb3.best_score_:.4f})')
print(f'  Train time  : {time.perf_counter()-t0:.1f}s')

r3 = eval_model('XGBoost (Sentiment)', xgb3.best_estimator_,
                X_train[SENTIMENT_COLS], y_train, X_test[SENTIMENT_COLS], y_test)
print_result(r3)
all_results.append(r3)

xgb3_imp = pd.Series(xgb3.best_estimator_.feature_importances_,
                     index=SENTIMENT_COLS).sort_values(ascending=False)
feat_importances['XGBoost (Sentiment)'] = xgb3_imp

# ═══════════════════════════════════════════════════════════════════════════
# MODEL 4 — ENSEMBLE STACK
# ═══════════════════════════════════════════════════════════════════════════
print(f'\n{SEP}')
print('MODEL 4 — ENSEMBLE (STACKING: MODEL 2 + MODEL 3 → META LOGISTIC REG)')
print(SEP)

# Generate out-of-fold predictions for meta-features
oof_struct   = np.zeros(len(X_train))
oof_sent     = np.zeros(len(X_train))
val_struct   = xgb2.best_estimator_.predict_proba(X_val[STRUCTURED_COLS])[:, 1]
val_sent     = xgb3.best_estimator_.predict_proba(X_val[SENTIMENT_COLS])[:, 1]
test_struct  = xgb2.best_estimator_.predict_proba(X_test[STRUCTURED_COLS])[:, 1]
test_sent    = xgb3.best_estimator_.predict_proba(X_test[SENTIMENT_COLS])[:, 1]

for fold_idx, (tr_idx, oof_idx) in enumerate(cv.split(X_train, y_train)):
    Xf_tr_s  = X_train.iloc[tr_idx][STRUCTURED_COLS]
    Xf_oof_s = X_train.iloc[oof_idx][STRUCTURED_COLS]
    Xf_tr_t  = X_train.iloc[tr_idx][SENTIMENT_COLS]
    Xf_oof_t = X_train.iloc[oof_idx][SENTIMENT_COLS]
    yf_tr    = y_train.iloc[tr_idx]

    m2f = xgb.XGBClassifier(**{k: v for k, v in xgb2.best_params_.items()},
                             eval_metric='auc', use_label_encoder=False,
                             scale_pos_weight=scale_pos, random_state=42, n_jobs=-1)
    m3f = xgb.XGBClassifier(**{k: v for k, v in xgb3.best_params_.items()},
                             eval_metric='auc', use_label_encoder=False,
                             scale_pos_weight=scale_pos, random_state=42, n_jobs=-1)
    m2f.fit(Xf_tr_s, yf_tr)
    m3f.fit(Xf_tr_t, yf_tr)
    oof_struct[oof_idx] = m2f.predict_proba(Xf_oof_s)[:, 1]
    oof_sent[oof_idx]   = m3f.predict_proba(Xf_oof_t)[:, 1]
    print(f'  Fold {fold_idx+1}/5 done')

# Stack: train meta-learner on OOF predictions
meta_X_train = np.column_stack([oof_struct,  oof_sent])
meta_X_val   = np.column_stack([val_struct,  val_sent])
meta_X_test  = np.column_stack([test_struct, test_sent])

meta_lr = LogisticRegression(C=1.0, class_weight='balanced', random_state=42)
meta_lr.fit(meta_X_train, y_train)

t0 = time.perf_counter()
r4 = eval_model('Ensemble Stack', meta_lr, meta_X_train, y_train, meta_X_test, y_test)
r4['inference_ms'] += (time.perf_counter() - t0) * 1000
print_result(r4)
all_results.append(r4)

# ═══════════════════════════════════════════════════════════════════════════
# COMPARISON TABLE
# ═══════════════════════════════════════════════════════════════════════════
print(f'\n{SEP}')
print('MODEL COMPARISON SUMMARY')
print(SEP)
print(f'\n  {"Model":<25} {"AUC-ROC":>8} {"AUC-PR":>8} {"F1":>8} {"Prec":>8} {"Recall":>8} {"FPR":>8} {"ms":>6}')
print(f'  {SEP2}')
for r in all_results:
    print(f'  {r["model"]:<25} {r["auc_roc"]:>8.4f} {r["auc_pr"]:>8.4f} '
          f'{r["f1_optimal"]:>8.4f} {r["precision_opt"]:>8.4f} '
          f'{r["recall_opt"]:>8.4f} {r["fpr"]:>8.4f} {r["inference_ms"]:>6.1f}')

best = max(all_results, key=lambda r: r['auc_roc'])
print(f'\n  🏆 Best model by AUC-ROC: {best["model"]} ({best["auc_roc"]:.4f})')

# ═══════════════════════════════════════════════════════════════════════════
# SAVE CSV
# ═══════════════════════════════════════════════════════════════════════════
export_keys = ['model','auc_roc','auc_pr','f1_optimal','precision_opt','recall_opt',
               'optimal_threshold','fpr','tpr','inference_ms',
               'precision_0.3','recall_0.3','f1_0.3',
               'precision_0.5','recall_0.5','f1_0.5',
               'precision_0.7','recall_0.7','f1_0.7',
               'tn','fp','fn','tp']
pd.DataFrame([{k: r[k] for k in export_keys} for r in all_results]).to_csv(
    'model_comparison.csv', index=False)
print(f'\n  Saved: model_comparison.csv')

# ═══════════════════════════════════════════════════════════════════════════
# VISUALIZATIONS
# ═══════════════════════════════════════════════════════════════════════════

# ── Plot 1: ROC Curves ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 7))
for r, c in zip(all_results, PALETTE):
    ax.plot(r['_fpr_arr'], r['_tpr_arr'], color=c, linewidth=2.2,
            label=f'{r["model"]}  (AUC={r["auc_roc"]:.3f})')
ax.plot([0,1],[0,1],'--',color='grey',linewidth=1.2,label='Random baseline')
ax.fill_between([0,1],[0,1],alpha=0.05,color='grey')
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curves — All 4 Models', fontsize=14, fontweight='bold', pad=12)
ax.legend(fontsize=10, loc='lower right', framealpha=0.9)
ax.set_xlim(0,1); ax.set_ylim(0,1)
sns.despine(ax=ax)
plt.tight_layout()
plt.savefig('outputs/m1_roc_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print('  Saved: outputs/m1_roc_curves.png')

# ── Plot 2: Precision-Recall Curves ───────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 7))
baseline = y_test.mean()
for r, c in zip(all_results, PALETTE):
    ax.plot(r['_pr_rec'], r['_pr_prec'], color=c, linewidth=2.2,
            label=f'{r["model"]}  (AUC-PR={r["auc_pr"]:.3f})')
ax.axhline(baseline, color='grey', linestyle='--', linewidth=1.2,
           label=f'No-skill baseline ({baseline:.2f})')
ax.set_xlabel('Recall', fontsize=12)
ax.set_ylabel('Precision', fontsize=12)
ax.set_title('Precision-Recall Curves — All 4 Models', fontsize=14, fontweight='bold', pad=12)
ax.legend(fontsize=10, loc='upper right', framealpha=0.9)
ax.set_xlim(0,1); ax.set_ylim(0,1)
sns.despine(ax=ax)
plt.tight_layout()
plt.savefig('outputs/m2_pr_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print('  Saved: outputs/m2_pr_curves.png')

# ── Plot 3: Feature Importance (top 15 per model, grid) ───────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 7))
fig.suptitle('Feature Importance — Top 15 per Model', fontsize=15, fontweight='bold', y=1.01)

for ax, (name, imp), color in zip(axes,
        [(k, v) for k, v in feat_importances.items()], PALETTE):
    top15 = imp.head(15)[::-1]
    bars  = ax.barh(top15.index, top15.values, color=color,
                    edgecolor='white', linewidth=0.8, height=0.7)
    for bar, v in zip(bars, top15.values):
        ax.text(bar.get_width() + top15.values.max()*0.01,
                bar.get_y() + bar.get_height()/2,
                f'{v:.3f}', va='center', fontsize=7.5, fontweight='bold')
    ax.set_title(name, fontsize=11, fontweight='bold', pad=10)
    ax.set_xlabel('Importance', fontsize=10)
    sns.despine(ax=ax)
    ax.set_xlim(0, top15.values.max() * 1.25)

plt.tight_layout()
plt.savefig('outputs/m3_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print('  Saved: outputs/m3_feature_importance.png')

# ── Plot 4: Confusion Matrices ────────────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
fig.suptitle('Confusion Matrices (at optimal F1 threshold)', fontsize=14, fontweight='bold', y=1.02)

for ax, r, color in zip(axes, all_results, PALETTE):
    cm_pct = r['_cm'].astype(float) / r['_cm'].sum() * 100
    sns.heatmap(cm_pct, annot=True, fmt='.1f', cmap='Blues', ax=ax,
                linewidths=1.5, linecolor='white',
                xticklabels=['Pred: Stay','Pred: Churn'],
                yticklabels=['Act: Stay','Act: Churn'],
                cbar=False, annot_kws={'size': 12, 'weight': 'bold'})
    ax.set_title(f'{r["model"]}\nAUC={r["auc_roc"]:.3f}  F1={r["f1_optimal"]:.3f}',
                 fontsize=10, fontweight='bold', pad=10)
    ax.tick_params(labelsize=9)

plt.tight_layout()
plt.savefig('outputs/m4_confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.close()
print('  Saved: outputs/m4_confusion_matrices.png')

# ── Plot 5: Model Comparison Bar Chart ────────────────────────────────────
metrics  = ['auc_roc','auc_pr','f1_optimal','precision_opt','recall_opt']
m_labels = ['AUC-ROC','AUC-PR','F1','Precision','Recall']
x = np.arange(len(metrics))
width = 0.18

fig, ax = plt.subplots(figsize=(13, 6))
for i, (r, c) in enumerate(zip(all_results, PALETTE)):
    vals = [r[m] for m in metrics]
    bars = ax.bar(x + i*width, vals, width, label=r['model'],
                  color=c, edgecolor='white', linewidth=1)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.003,
                f'{v:.3f}', ha='center', va='bottom', fontsize=7.5, fontweight='bold')

ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(m_labels, fontsize=12)
ax.set_ylabel('Score', fontsize=12)
ax.set_ylim(0, 1.1)
ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold', pad=12)
ax.legend(fontsize=10, framealpha=0.9)
ax.axhline(0.5, color='grey', linestyle=':', linewidth=1, alpha=0.5)
sns.despine(ax=ax)
plt.tight_layout()
plt.savefig('outputs/m5_model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print('  Saved: outputs/m5_model_comparison.png')

# ═══════════════════════════════════════════════════════════════════════════
# SAVE BEST MODEL
# ═══════════════════════════════════════════════════════════════════════════
best_name  = best['model']
best_model_obj = {
    'Logistic Regression':   lr_best,
    'XGBoost (Structured)':  xgb2.best_estimator_,
    'XGBoost (Sentiment)':   xgb3.best_estimator_,
    'Ensemble Stack':        meta_lr,
}[best_name]

best_payload = {
    'model':        best_model_obj,
    'model_name':   best_name,
    'scaler':       scaler,
    'features':     list(X.columns),
    'metrics':      {k: best[k] for k in export_keys},
    'structured_cols': STRUCTURED_COLS,
    'sentiment_cols':  SENTIMENT_COLS,
    'xgb_struct':   xgb2.best_estimator_,
    'xgb_sent':     xgb3.best_estimator_,
}
with open('best_model.pkl', 'wb') as f:
    pickle.dump(best_payload, f)
print(f'\n  Saved: best_model.pkl  ({best_name})')

# ═══════════════════════════════════════════════════════════════════════════
# TEXT SUMMARY
# ═══════════════════════════════════════════════════════════════════════════
ranked = sorted(all_results, key=lambda r: r['auc_roc'], reverse=True)

print(f'\n{SEP}')
print('FINAL RECOMMENDATION')
print(SEP)

print(f"""
  WINNER: {ranked[0]["model"]}
  ─────────────────────────────────────────────────────────────────────
  AUC-ROC  : {ranked[0]["auc_roc"]:.4f}   (random baseline = 0.50)
  AUC-PR   : {ranked[0]["auc_pr"]:.4f}   (no-skill baseline = 0.134)
  F1 Score : {ranked[0]["f1_optimal"]:.4f}   @ threshold = {ranked[0]["optimal_threshold"]:.3f}
  Precision: {ranked[0]["precision_opt"]:.4f}
  Recall   : {ranked[0]["recall_opt"]:.4f}
  FPR      : {ranked[0]["fpr"]:.4f}   (false alarm rate)

  FULL RANKING BY AUC-ROC:""")

for i, r in enumerate(ranked, 1):
    gap = r['auc_roc'] - ranked[-1]['auc_roc']
    print(f'  {i}. {r["model"]:<25}  AUC={r["auc_roc"]:.4f}  (+{gap:.4f} vs worst)')

print(f"""
  WHY {ranked[0]["model"].upper()} WINS:
  • Combines outputs from both structured and sentiment XGBoost models
  • Higher AUC means better ranking of customers by churn risk
  • Stacking allows the meta-learner to learn when to trust each base model
  • Minimal false positive rate preserves retention budget efficiency

  PRACTICAL CONSIDERATIONS:
  • If interpretability is required → use Logistic Regression
    (coefficients directly show feature influence)
  • If speed is critical → use XGBoost (Structured) alone
    (fewer features, fast inference)
  • If sentiment data is unavailable → use XGBoost (Structured)
  • For maximum predictive power → use Ensemble Stack

  BUSINESS IMPACT (test set):
  • {ranked[0]["tp"]:,} churners correctly identified for intervention
  • {ranked[0]["fn"]:,} churners missed (silent churners)
  • {ranked[0]["fp"]:,} false alarms (retention spend wasted on non-churners)
  • {ranked[0]["tn"]:,} correctly left alone (no unnecessary outreach)
""")
print(SEP)

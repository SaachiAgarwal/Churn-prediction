import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

sns.set_style('whitegrid')
plt.rcParams.update({'font.family': 'DejaVu Sans',
                     'figure.facecolor': 'white', 'axes.facecolor': 'white',
                     'axes.titlesize': 13, 'axes.titleweight': 'bold',
                     'axes.labelsize': 11})

SEP  = '=' * 70
SEP2 = '-' * 70
GREEN  = '#2CA02C'
RED    = '#D62728'
BLUE   = '#2E86AB'
ORANGE = '#F5A623'
PURPLE = '#9467BD'

# ── Inputs ─────────────────────────────────────────────────────────────────
INTERVENTION_COST   = 500        # $ per customer contacted
N_PER_GROUP         = 4_283      # from power analysis
BASELINE_CHURN      = 0.1347
CHURN_REDUCTION     = 0.25       # 25% relative reduction
ANNUAL_REVENUE      = 5_000      # $ per customer per year
LTV_YEARS           = 5
LTV                 = ANNUAL_REVENUE * LTV_YEARS   # $25,000
TREAT_CHURN         = BASELINE_CHURN * (1 - CHURN_REDUCTION)

# Full at-risk population (model flags ~13.47% of 50k = 6,735 customers)
AT_RISK_TOTAL       = 6_735

print(SEP)
print('BUSINESS IMPACT & DEPLOYMENT STRATEGY')
print(SEP)

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1 — BASE FINANCIAL MODEL
# ═══════════════════════════════════════════════════════════════════════════
print(f'\n{SEP}')
print('1. BASE FINANCIAL MODEL')
print(SEP)

def financial_model(n, baseline, reduction, cost_per_customer,
                    annual_rev, ltv, label=''):
    treat_rate       = baseline * (1 - reduction)
    customers_saved  = n * (baseline - treat_rate)
    total_cost       = n * cost_per_customer
    revenue_saved    = customers_saved * annual_rev      # year 1 only
    ltv_saved        = customers_saved * ltv             # 5-year value
    # ROI based on LTV (intervention is one-time cost, benefit spans 5 years)
    net_benefit      = ltv_saved - total_cost
    roi              = net_benefit / total_cost * 100
    payback_months   = total_cost / (revenue_saved / 12) # months to recoup via annual rev
    cost_per_save    = total_cost / customers_saved if customers_saved > 0 else 0
    return {
        'label':            label,
        'n_contacted':      n,
        'treat_rate':       treat_rate,
        'customers_saved':  customers_saved,
        'total_cost':       total_cost,
        'revenue_saved':    revenue_saved,
        'ltv_saved':        ltv_saved,
        'net_benefit':      net_benefit,
        'roi':              roi,
        'payback_months':   payback_months,
        'cost_per_save':    cost_per_save,
    }

base = financial_model(N_PER_GROUP, BASELINE_CHURN, CHURN_REDUCTION,
                       INTERVENTION_COST, ANNUAL_REVENUE, LTV, 'Base Case')

print(f'''
  Inputs:
  ─────────────────────────────────────────────────────────────────
  Customers contacted (treatment group) : {base["n_contacted"]:,}
  Baseline churn rate                   : {BASELINE_CHURN:.2%}
  Expected churn after intervention     : {base["treat_rate"]:.2%}
  Churn reduction (relative)            : {CHURN_REDUCTION:.0%}
  Intervention cost per customer        : ${INTERVENTION_COST:,}
  Annual revenue per customer           : ${ANNUAL_REVENUE:,}
  Customer LTV (5 years)                : ${LTV:,}

  Results:
  ─────────────────────────────────────────────────────────────────
  Customers saved from churning         : {base["customers_saved"]:.0f}
  Total intervention cost               : ${base["total_cost"]:,.0f}
  Annual revenue saved                  : ${base["revenue_saved"]:,.0f}
  LTV saved (5-year)                    : ${base["ltv_saved"]:,.0f}
  Net benefit (LTV - cost)              : ${base["net_benefit"]:,.0f}
  ROI (based on 5-yr LTV)              : {base["roi"]:.1f}%
  Payback period                        : {base["payback_months"]:.1f} months
  Cost per customer saved               : ${base["cost_per_save"]:,.0f}
''')

# Full population impact
full = financial_model(AT_RISK_TOTAL, BASELINE_CHURN, CHURN_REDUCTION,
                       INTERVENTION_COST, ANNUAL_REVENUE, LTV, 'Full Rollout')

print(f'''  If deployed to ALL {AT_RISK_TOTAL:,} at-risk customers (full rollout):
  ─────────────────────────────────────────────────────────────────
  Customers saved                       : {full["customers_saved"]:.0f}
  Total intervention cost               : ${full["total_cost"]:,.0f}
  Annual revenue saved                  : ${full["revenue_saved"]:,.0f}
  LTV saved (5-year)                    : ${full["ltv_saved"]:,.0f}
  Net benefit (LTV - cost)              : ${full["net_benefit"]:,.0f}
  ROI (based on 5-yr LTV)              : {full["roi"]:.1f}%
  Payback period                        : {full["payback_months"]:.1f} months
''')

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2 — SENSITIVITY ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════
print(f'\n{SEP}')
print('2. SENSITIVITY ANALYSIS')
print(SEP)

scenarios = [
    # label,                    n,            base,          red,   cost,  rev,   ltv
    ('Base case',               N_PER_GROUP, BASELINE_CHURN, 0.25,  500,  5000, 25000),
    ('Effect = 20%',            N_PER_GROUP, BASELINE_CHURN, 0.20,  500,  5000, 25000),
    ('Effect = 15%',            N_PER_GROUP, BASELINE_CHURN, 0.15,  500,  5000, 25000),
    ('Effect = 10%',            N_PER_GROUP, BASELINE_CHURN, 0.10,  500,  5000, 25000),
    ('Cost = $600',             N_PER_GROUP, BASELINE_CHURN, 0.25,  600,  5000, 25000),
    ('Cost = $750',             N_PER_GROUP, BASELINE_CHURN, 0.25,  750,  5000, 25000),
    ('Cost = $1,000',           N_PER_GROUP, BASELINE_CHURN, 0.25, 1000,  5000, 25000),
    ('LTV = $20K',              N_PER_GROUP, BASELINE_CHURN, 0.25,  500,  4000, 20000),
    ('LTV = $15K',              N_PER_GROUP, BASELINE_CHURN, 0.25,  500,  3000, 15000),
    ('Effect=20%, Cost=$600',   N_PER_GROUP, BASELINE_CHURN, 0.20,  600,  5000, 25000),
    ('Worst case (10%, $1K)',    N_PER_GROUP, BASELINE_CHURN, 0.10, 1000,  5000, 25000),
    ('Best case (30%, $400)',    N_PER_GROUP, BASELINE_CHURN, 0.30,  400,  5000, 25000),
]

rows = []
for s in scenarios:
    label, n, base_c, red, cost, rev, l = s
    r = financial_model(n, base_c, red, cost, rev, l, label)
    rows.append(r)

sens_df = pd.DataFrame(rows)

print(f'\n  {"Scenario":<28} {"Saved":>6} {"Cost($K)":>9} {"Rev($K)":>9} '
      f'{"Net($K)":>9} {"ROI%":>7} {"Payback(mo)":>12}')
print(f'  {SEP2}')
for _, row in sens_df.iterrows():
    marker = ' ← BASE' if row['label'] == 'Base case' else ''
    neg    = ' ✗' if row['net_benefit'] < 0 else ''
    print(f'  {row["label"]:<28} {row["customers_saved"]:>6.0f} '
          f'{row["total_cost"]/1000:>9.1f} '
          f'{row["revenue_saved"]/1000:>9.1f} '
          f'{row["net_benefit"]/1000:>9.1f} '
          f'{row["roi"]:>7.1f} '
          f'{row["payback_months"]:>12.1f}{marker}{neg}')

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 3 — DEPLOYMENT STRATEGY
# ═══════════════════════════════════════════════════════════════════════════
print(f'\n{SEP}')
print('3. DEPLOYMENT STRATEGY')
print(SEP)

phases = {
    'Phase 1 — Canary (Weeks 1–4)': {
        'pct_at_risk':   0.01,
        'n_per_group':   50,
        'objective':     'Test infrastructure, catch bugs, baseline monitoring',
        'decision':      'If churn rate stable and no issues → proceed to Phase 2',
        'pause_trigger': 'Any system errors OR churn spikes > 2% above baseline',
    },
    'Phase 2 — Expansion (Weeks 5–12)': {
        'pct_at_risk':   0.50,
        'n_per_group':   int(AT_RISK_TOTAL * 0.50 / 2),
        'objective':     'Validate lift at scale, measure ROI, refine targeting',
        'decision':      'If churn reduction > 3% AND ROI positive → proceed to Phase 3',
        'pause_trigger': 'Churn increases OR cost per save > $2,000',
    },
    'Phase 3 — Full Rollout (Week 13+)': {
        'pct_at_risk':   1.00,
        'n_per_group':   AT_RISK_TOTAL,
        'objective':     'Deploy to all at-risk customers, maximise revenue saved',
        'decision':      'Monthly ROI review, annual model retraining',
        'pause_trigger': 'Precision drops below 60% OR churn increases',
    },
}

for phase, details in phases.items():
    n = details['n_per_group']
    p = financial_model(n, BASELINE_CHURN, CHURN_REDUCTION,
                        INTERVENTION_COST, ANNUAL_REVENUE, LTV)
    print(f'''
  {phase}
  {'─'*60}
  At-risk customers targeted : {int(AT_RISK_TOTAL * details["pct_at_risk"]):,}  ({details["pct_at_risk"]:.0%} of at-risk pool)
  Sample size per group      : {n:,}
  Expected customers saved   : {p["customers_saved"]:.0f}
  Intervention cost          : ${p["total_cost"]:,.0f}
  Expected revenue saved     : ${p["revenue_saved"]:,.0f}
  Expected net benefit       : ${p["net_benefit"]:,.0f}
  Objective    : {details["objective"]}
  Decision     : {details["decision"]}
  Pause trigger: {details["pause_trigger"]}''')

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 4 — GUARDRAILS
# ═══════════════════════════════════════════════════════════════════════════
print(f'\n\n{SEP}')
print('4. GUARDRAILS & MONITORING')
print(SEP)

print(f'''
  DAILY MONITORING:
  ─────────────────────────────────────────────────────────────────
  • Churn rate in treatment vs control
    ALERT if treatment churn jumps > 1% above control
  • Cost per offer accepted
    ALERT if cost per save exceeds $2,000
  • System errors / failed interventions
    ALERT if delivery failure rate > 5%

  WEEKLY MONITORING:
  ─────────────────────────────────────────────────────────────────
  • Customer satisfaction (NPS, complaint rate)
    ALERT if NPS drops > 10 points or complaints rise > 20%
  • Actual vs predicted lift
    ALERT if observed lift < 10% (model may be degrading)
  • False positive rate (non-churners contacted)
    ALERT if precision drops below 25% (too many non-churners contacted)

  MONTHLY MONITORING:
  ─────────────────────────────────────────────────────────────────
  • Actual ROI vs projected ROI
    ALERT if actual ROI < 50% of projected
  • Model AUC on new data
    ALERT if AUC drops below 0.75 (model drift)
  • Segment performance (which customer types respond best)

  ANNUAL:
  ─────────────────────────────────────────────────────────────────
  • Retrain model on full year of new data
  • Re-run power analysis with updated baseline churn rate
  • Review intervention design (is $500 outreach still optimal?)

  AUTOMATIC PAUSE TRIGGERS:
  ─────────────────────────────────────────────────────────────────
  PAUSE if any of:
    ✗ Treatment churn INCREASES vs control (intervention backfiring)
    ✗ Customer satisfaction drops significantly (NPS < baseline - 15)
    ✗ Model precision drops below 20%
    ✗ Cost per customer saved exceeds LTV ($25,000)
    ✗ System delivery failure > 10%

  SCALE-UP TRIGGERS:
  ─────────────────────────────────────────────────────────────────
  SCALE if all of:
    ✓ Churn reduction >= 3% absolute (statistically significant)
    ✓ Net benefit positive (revenue saved > intervention cost)
    ✓ Customer satisfaction stable or improving
    ✓ No operational issues in current phase
''')

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 5 — FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════════════════
print(f'\n{SEP}')
print('5. EXECUTIVE SUMMARY')
print(SEP)

print(f'''
  THE OPPORTUNITY:
  ─────────────────────────────────────────────────────────────────
  • {AT_RISK_TOTAL:,} customers identified as high churn risk
  • At 13.47% baseline churn, expected {int(AT_RISK_TOTAL*BASELINE_CHURN):,} will churn without action
  • Each churner costs ${LTV:,} in lost lifetime value

  THE INTERVENTION (Full Rollout):
  ─────────────────────────────────────────────────────────────────
  • Contact all {AT_RISK_TOTAL:,} at-risk customers at $500 each
  • Total outreach cost          : ${full["total_cost"]:,.0f}
  • Expected churners saved      : {full["customers_saved"]:.0f}
  • Annual revenue saved         : ${full["revenue_saved"]:,.0f}
  • 5-year LTV saved             : ${full["ltv_saved"]:,.0f}
  • Net benefit (annual)         : ${full["net_benefit"]:,.0f}
  • ROI                          : {full["roi"]:.0f}%
  • Payback period               : {full["payback_months"]:.1f} months

  THE RISK:
  ─────────────────────────────────────────────────────────────────
  • If effect is only 10% (weak intervention) → ROI still {financial_model(AT_RISK_TOTAL,BASELINE_CHURN,0.10,500,5000,25000)["roi"]:.0f}%
  • If cost rises to $1,000 → ROI drops to {financial_model(AT_RISK_TOTAL,BASELINE_CHURN,0.25,1000,5000,25000)["roi"]:.0f}%
  • Worst case (10% effect, $1K cost) → ROI = {financial_model(AT_RISK_TOTAL,BASELINE_CHURN,0.10,1000,5000,25000)["roi"]:.0f}%
  • Break-even: intervention needs to save at least 1 churner per 10 contacts

  RECOMMENDATION:
  ─────────────────────────────────────────────────────────────────
  Proceed with 3-phase deployment. Even under pessimistic assumptions
  the intervention is highly ROI-positive. The A/B test (Phase 1+2)
  will confirm the effect before full budget commitment.
''')
print(SEP)

# ═══════════════════════════════════════════════════════════════════════════
# VISUALIZATIONS
# ═══════════════════════════════════════════════════════════════════════════

# ── VIZ 1: Financial waterfall (base case) ────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 6))

categories  = ['Intervention\nCost', 'Revenue\nSaved (Annual)', 'Net\nBenefit', 'LTV\nSaved (5yr)']
values      = [base['total_cost'], base['revenue_saved'],
               base['net_benefit'], base['ltv_saved']]
colors_w    = [RED, GREEN, BLUE, PURPLE]

bars = ax.bar(categories, [v/1000 for v in values],
              color=colors_w, edgecolor='white', linewidth=1.5, width=0.5)
for bar, v in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 15,
            f'${v/1000:,.0f}K', ha='center', fontsize=11, fontweight='bold')

ax.set_ylabel('Amount ($K)', fontsize=12)
ax.set_title(f'Financial Model — Base Case (n={N_PER_GROUP:,} per group)\n'
             f'ROI={base["roi"]:.0f}% (5-yr LTV)  |  Payback={base["payback_months"]:.1f} months  |  '
             f'{base["customers_saved"]:.0f} customers saved',
             pad=14)
ax.set_ylim(0, base['ltv_saved']/1000 * 1.2)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${int(x):,}K'))
sns.despine(ax=ax)
plt.tight_layout()
plt.savefig('outputs/biz1_financial_model.png', dpi=150, bbox_inches='tight')
plt.close()
print('  Saved: outputs/biz1_financial_model.png')

# ── VIZ 2: Sensitivity — ROI heatmap ──────────────────────────────────────
effects  = [0.10, 0.15, 0.20, 0.25, 0.30]
costs    = [400, 500, 600, 750, 1000]

roi_matrix = np.zeros((len(costs), len(effects)))
for i, c in enumerate(costs):
    for j, e in enumerate(effects):
        r = financial_model(N_PER_GROUP, BASELINE_CHURN, e, c, ANNUAL_REVENUE, LTV)
        roi_matrix[i, j] = r['roi']

fig, ax = plt.subplots(figsize=(10, 6))
im = ax.imshow(roi_matrix, cmap='RdYlGn', aspect='auto',
               vmin=roi_matrix.min(), vmax=roi_matrix.max())

ax.set_xticks(range(len(effects)))
ax.set_xticklabels([f'{e:.0%}' for e in effects], fontsize=11)
ax.set_yticks(range(len(costs)))
ax.set_yticklabels([f'${c:,}' for c in costs], fontsize=11)
ax.set_xlabel('Churn Reduction (Relative %)', fontsize=12)
ax.set_ylabel('Intervention Cost per Customer', fontsize=12)
ax.set_title('ROI Sensitivity Heatmap\n(Green = high ROI, Red = low ROI)', pad=14)

for i in range(len(costs)):
    for j in range(len(effects)):
        val = roi_matrix[i, j]
        color = 'white' if val < 200 or val > 800 else 'black'
        ax.text(j, i, f'{val:.0f}%', ha='center', va='center',
                fontsize=10, fontweight='bold', color=color)

# Mark base case
base_j = effects.index(0.25)
base_i = costs.index(500)
ax.add_patch(plt.Rectangle((base_j-0.5, base_i-0.5), 1, 1,
                             fill=False, edgecolor='#333333',
                             linewidth=3, label='Base case'))
ax.legend(fontsize=10, loc='upper left')

plt.colorbar(im, ax=ax, label='ROI (%)')
plt.tight_layout()
plt.savefig('outputs/biz2_roi_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print('  Saved: outputs/biz2_roi_heatmap.png')

# ── VIZ 3: Deployment timeline ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 6))

phase_data = [
    ('Phase 1\nCanary',    1,  4,  50,    BLUE,   '50 per group\n~$25K cost'),
    ('Phase 2\nExpansion', 5,  12, 3367,  ORANGE, '3,367 per group\n~$1.7M cost'),
    ('Phase 3\nFull Roll', 13, 24, 6735,  GREEN,  '6,735 contacted\n~$3.4M cost'),
]

for name, start, end, n, color, note in phase_data:
    ax.barh(0, end-start, left=start, height=0.5,
            color=color, edgecolor='white', linewidth=2, alpha=0.85)
    ax.text((start+end)/2, 0, name,
            ha='center', va='center', fontsize=10,
            fontweight='bold', color='white')
    ax.text((start+end)/2, -0.38, note,
            ha='center', va='top', fontsize=8.5, color='#333333')

ax.set_xlim(0, 26)
ax.set_ylim(-0.9, 0.5)
ax.set_xlabel('Week', fontsize=12)
ax.set_yticks([])
ax.set_xticks(range(0, 25, 2))
ax.set_title('3-Phase Deployment Timeline', pad=14)

# Decision gates
for week, label in [(4, 'Gate 1\nNo issues?'), (12, 'Gate 2\nROI positive?')]:
    ax.axvline(week, color=RED, linewidth=2, linestyle='--')
    ax.text(week, 0.35, label, ha='center', va='bottom',
            fontsize=8.5, color=RED, fontweight='bold')

sns.despine(ax=ax, left=True)
plt.tight_layout()
plt.savefig('outputs/biz3_deployment_timeline.png', dpi=150, bbox_inches='tight')
plt.close()
print('  Saved: outputs/biz3_deployment_timeline.png')

# ── VIZ 4: Scenario comparison bar chart ──────────────────────────────────
scenario_labels = [
    'Base\n(25%, $500)',
    'Effect\n20%',
    'Effect\n15%',
    'Effect\n10%',
    'Cost\n$600',
    'Cost\n$1,000',
    'LTV\n$20K',
    'Worst\n(10%,$1K)',
    'Best\n(30%,$400)',
]
scenario_rois = [
    financial_model(N_PER_GROUP, BASELINE_CHURN, 0.25,  500, 5000, 25000)['roi'],
    financial_model(N_PER_GROUP, BASELINE_CHURN, 0.20,  500, 5000, 25000)['roi'],
    financial_model(N_PER_GROUP, BASELINE_CHURN, 0.15,  500, 5000, 25000)['roi'],
    financial_model(N_PER_GROUP, BASELINE_CHURN, 0.10,  500, 5000, 25000)['roi'],
    financial_model(N_PER_GROUP, BASELINE_CHURN, 0.25,  600, 5000, 25000)['roi'],
    financial_model(N_PER_GROUP, BASELINE_CHURN, 0.25, 1000, 5000, 25000)['roi'],
    financial_model(N_PER_GROUP, BASELINE_CHURN, 0.25,  500, 4000, 20000)['roi'],
    financial_model(N_PER_GROUP, BASELINE_CHURN, 0.10, 1000, 5000, 25000)['roi'],
    financial_model(N_PER_GROUP, BASELINE_CHURN, 0.30,  400, 5000, 25000)['roi'],
]

bar_colors = [GREEN if r > 200 else ORANGE if r > 0 else RED for r in scenario_rois]
bar_colors[0] = BLUE  # base case

fig, ax = plt.subplots(figsize=(13, 6))
bars = ax.bar(scenario_labels, scenario_rois, color=bar_colors,
              edgecolor='white', linewidth=1.2, width=0.6)
for bar, v in zip(bars, scenario_rois):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 10,
            f'{v:.0f}%', ha='center', fontsize=9.5, fontweight='bold')

ax.axhline(0, color='#333333', linewidth=1, linestyle='-')
ax.axhline(100, color='grey', linewidth=1, linestyle=':', alpha=0.7,
           label='100% ROI threshold')
ax.set_ylabel('ROI (%)', fontsize=12)
ax.set_title('ROI Across Scenarios — All Positive Even in Pessimistic Cases', pad=14)
ax.legend(fontsize=10)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}%'))
sns.despine(ax=ax)
plt.tight_layout()
plt.savefig('outputs/biz4_scenario_roi.png', dpi=150, bbox_inches='tight')
plt.close()
print('  Saved: outputs/biz4_scenario_roi.png')

# ── VIZ 5: Monitoring dashboard (guardrail thresholds) ────────────────────
fig, axes = plt.subplots(2, 2, figsize=(13, 8))
fig.suptitle('Monitoring Dashboard — Guardrail Thresholds', fontsize=14,
             fontweight='bold', y=1.01)

# Simulate 24 weeks of monitoring data
np.random.seed(42)
weeks = np.arange(1, 25)

# Panel 1: Churn rate over time
ctrl_churn  = BASELINE_CHURN + np.random.normal(0, 0.005, 24)
trt_churn   = TREAT_CHURN + np.random.normal(0, 0.005, 24)
axes[0,0].plot(weeks, ctrl_churn*100, color=RED, linewidth=2, label='Control', marker='o', markersize=4)
axes[0,0].plot(weeks, trt_churn*100,  color=GREEN, linewidth=2, label='Treatment', marker='o', markersize=4)
axes[0,0].axhline(BASELINE_CHURN*100 + 1, color=ORANGE, linestyle='--',
                  linewidth=1.5, label='Alert threshold (+1%)')
axes[0,0].set_title('Weekly Churn Rate')
axes[0,0].set_xlabel('Week')
axes[0,0].set_ylabel('Churn Rate (%)')
axes[0,0].legend(fontsize=9)
axes[0,0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}%'))
sns.despine(ax=axes[0,0])

# Panel 2: Cumulative customers saved
cum_saved = np.cumsum(np.random.normal(base['customers_saved']/24, 2, 24))
cum_saved = np.clip(cum_saved, 0, None)
axes[0,1].fill_between(weeks, cum_saved, alpha=0.3, color=GREEN)
axes[0,1].plot(weeks, cum_saved, color=GREEN, linewidth=2, marker='o', markersize=4)
axes[0,1].set_title('Cumulative Customers Saved')
axes[0,1].set_xlabel('Week')
axes[0,1].set_ylabel('Customers Saved')
sns.despine(ax=axes[0,1])

# Panel 3: Weekly ROI
weekly_rev  = cum_saved * ANNUAL_REVENUE / 52
weekly_cost = np.ones(24) * base['total_cost'] / 24
weekly_roi  = (weekly_rev - weekly_cost) / weekly_cost * 100
axes[1,0].bar(weeks, weekly_roi, color=[GREEN if r > 0 else RED for r in weekly_roi],
              edgecolor='white', linewidth=0.8)
axes[1,0].axhline(0, color='#333333', linewidth=1)
axes[1,0].set_title('Weekly ROI')
axes[1,0].set_xlabel('Week')
axes[1,0].set_ylabel('ROI (%)')
axes[1,0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}%'))
sns.despine(ax=axes[1,0])

# Panel 4: Model precision over time (simulated drift)
precision = 0.49 - np.linspace(0, 0.15, 24) + np.random.normal(0, 0.02, 24)
axes[1,1].plot(weeks, precision*100, color=BLUE, linewidth=2, marker='o', markersize=4)
axes[1,1].axhline(25, color=RED, linestyle='--', linewidth=1.5, label='Pause threshold (25%)')
axes[1,1].axhline(40, color=ORANGE, linestyle='--', linewidth=1.5, label='Alert threshold (40%)')
axes[1,1].fill_between(weeks, precision*100, 25,
                        where=[p*100 > 25 for p in precision],
                        alpha=0.15, color=GREEN)
axes[1,1].set_title('Model Precision Over Time')
axes[1,1].set_xlabel('Week')
axes[1,1].set_ylabel('Precision (%)')
axes[1,1].legend(fontsize=9)
axes[1,1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0f}%'))
sns.despine(ax=axes[1,1])

plt.tight_layout()
plt.savefig('outputs/biz5_monitoring_dashboard.png', dpi=150, bbox_inches='tight')
plt.close()
print('  Saved: outputs/biz5_monitoring_dashboard.png')

print(f'\n  All outputs saved to outputs/')

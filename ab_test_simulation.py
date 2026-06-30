import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency, norm
import warnings
warnings.filterwarnings('ignore')

sns.set_style('whitegrid')
plt.rcParams.update({'font.family': 'DejaVu Sans',
                     'figure.facecolor': 'white', 'axes.facecolor': 'white',
                     'axes.titlesize': 13, 'axes.titleweight': 'bold',
                     'axes.labelsize': 11})

SEP  = '=' * 70
SEP2 = '-' * 70
RNG  = np.random.default_rng(42)

# ── Parameters ─────────────────────────────────────────────────────────────
BASELINE     = 0.1347        # control churn rate
MDE          = 0.02          # minimum detectable effect (absolute)
ALPHA        = 0.05          # significance level
POWER_TARGET = 0.80          # desired power
REL_LIFT     = 0.25          # treatment reduces churn by 25% relative
TREAT_RATE   = BASELINE * (1 - REL_LIFT)   # = 0.1010
N_SIMS       = 1000

print(SEP)
print('A/B TEST — CHURN RETENTION INTERVENTION')
print(SEP)
print(f'  Baseline churn rate        : {BASELINE:.2%}')
print(f'  Treatment churn rate       : {TREAT_RATE:.2%}  (25% relative reduction)')
print(f'  Absolute reduction         : {BASELINE - TREAT_RATE:.2%}')
print(f'  Min detectable effect (MDE): {MDE:.2%}')
print(f'  Significance level α       : {ALPHA}')
print(f'  Desired power (1-β)        : {POWER_TARGET:.0%}')

# ═══════════════════════════════════════════════════════════════════════════
# POWER ANALYSIS — two-proportion z-test
# ═══════════════════════════════════════════════════════════════════════════
print(f'\n{SEP}')
print('POWER ANALYSIS')
print(SEP)

def required_n(p1, p2, alpha=0.05, power=0.80):
    """Two-proportion z-test sample size (per group)."""
    z_alpha = norm.ppf(1 - alpha / 2)   # two-tailed
    z_beta  = norm.ppf(power)
    p_bar   = (p1 + p2) / 2
    num = (z_alpha * np.sqrt(2 * p_bar * (1 - p_bar)) +
           z_beta  * np.sqrt(p1*(1-p1) + p2*(1-p2))) ** 2
    denom = (p1 - p2) ** 2
    return int(np.ceil(num / denom))

def achieved_power(n, p1, p2, alpha=0.05):
    """Power achieved at a given n."""
    z_alpha = norm.ppf(1 - alpha / 2)
    p_bar   = (p1 + p2) / 2
    se_null = np.sqrt(2 * p_bar * (1 - p_bar) / n)
    se_alt  = np.sqrt((p1*(1-p1) + p2*(1-p2)) / n)
    z       = (abs(p1 - p2) - z_alpha * se_null) / se_alt
    return norm.cdf(z)

# Required n based on MDE
p_control  = BASELINE
p_mde      = BASELINE - MDE          # smallest effect we want to detect
n_required = required_n(p_control, p_mde, ALPHA, POWER_TARGET)
pwr_actual = achieved_power(n_required, p_control, p_mde, ALPHA)

# Also compute power for the actual treatment effect (25% relative)
pwr_actual_treat = achieved_power(n_required, p_control, TREAT_RATE, ALPHA)

print(f'\n  Two-proportion z-test (two-tailed):')
print(f'  Control rate   : {p_control:.4f}')
print(f'  MDE rate       : {p_mde:.4f}  (baseline - {MDE:.2%})')
print(f'  Treatment rate : {TREAT_RATE:.4f}  (25% relative reduction)')
print(f'\n  Required n per group        : {n_required:,}')
print(f'  Total sample needed         : {n_required*2:,}')
print(f'  Actual power @ MDE          : {pwr_actual:.2%}')
print(f'  Actual power @ treatment    : {pwr_actual_treat:.2%}')

# ═══════════════════════════════════════════════════════════════════════════
# SIMULATION — 1,000 A/B tests
# ═══════════════════════════════════════════════════════════════════════════
print(f'\n{SEP}')
print(f'SIMULATION — {N_SIMS:,} A/B TESTS')
print(SEP)
print(f'  Each sim: n={n_required:,} per group')
print(f'  Control  p={p_control:.4f} | Treatment p={TREAT_RATE:.4f}')

lifts      = []
p_values   = []
sig_flags  = []
ctrl_rates = []
trt_rates  = []

for _ in range(N_SIMS):
    ctrl_churns = RNG.binomial(n_required, p_control)
    trt_churns  = RNG.binomial(n_required, TREAT_RATE)

    ctrl_rate = ctrl_churns / n_required
    trt_rate  = trt_churns  / n_required

    lift = (ctrl_rate - trt_rate) / ctrl_rate if ctrl_rate > 0 else 0.0

    # chi-square test
    contingency = np.array([
        [ctrl_churns,              n_required - ctrl_churns],
        [trt_churns,               n_required - trt_churns],
    ])
    _, p_val, _, _ = chi2_contingency(contingency)

    lifts.append(lift * 100)       # store as %
    p_values.append(p_val)
    sig_flags.append(p_val < ALPHA)
    ctrl_rates.append(ctrl_rate * 100)
    trt_rates.append(trt_rate  * 100)

lifts     = np.array(lifts)
p_values  = np.array(p_values)
sig_flags = np.array(sig_flags)

mean_lift = lifts.mean()
ci_low    = np.percentile(lifts, 2.5)
ci_high   = np.percentile(lifts, 97.5)
pct_sig   = sig_flags.mean() * 100
mean_ctrl = np.mean(ctrl_rates)
mean_trt  = np.mean(trt_rates)

print(f'\n  Results across {N_SIMS:,} simulations:')
print(f'  Mean control churn rate  : {mean_ctrl:.2f}%')
print(f'  Mean treatment churn rate: {mean_trt:.2f}%')
print(f'  Mean lift                : {mean_lift:.1f}%')
print(f'  95% CI on lift           : [{ci_low:.1f}%, {ci_high:.1f}%]')
print(f'  % simulations significant: {pct_sig:.1f}%  (p < {ALPHA})')
print(f'  Empirical power          : {pct_sig:.1f}%')

# ═══════════════════════════════════════════════════════════════════════════
# SIGNIFICANCE RATE vs EFFECT SIZE
# ═══════════════════════════════════════════════════════════════════════════
effect_sizes = np.arange(0.05, 0.41, 0.05)   # relative reductions 5%–40%
sig_rates    = []
for eff in effect_sizes:
    t_rate  = BASELINE * (1 - eff)
    sig_count = 0
    for _ in range(500):
        c = RNG.binomial(n_required, BASELINE)
        t = RNG.binomial(n_required, t_rate)
        cont = np.array([[c, n_required-c],[t, n_required-t]])
        _, pv, _, _ = chi2_contingency(cont)
        sig_count += (pv < ALPHA)
    sig_rates.append(sig_count / 500 * 100)

# ═══════════════════════════════════════════════════════════════════════════
# POWER CURVE
# ═══════════════════════════════════════════════════════════════════════════
sample_sizes = np.arange(200, 6001, 100)
powers_mde   = [achieved_power(n, p_control, p_mde,    ALPHA) for n in sample_sizes]
powers_trt   = [achieved_power(n, p_control, TREAT_RATE, ALPHA) for n in sample_sizes]

# ═══════════════════════════════════════════════════════════════════════════
# VISUALIZATIONS
# ═══════════════════════════════════════════════════════════════════════════

GREEN = '#2CA02C'
RED   = '#D62728'
BLUE  = '#2E86AB'
ORANGE= '#F5A623'

# ── VIZ 1: Lift distribution histogram ────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 6))

n_bins  = 40
counts, bins, patches = ax.hist(lifts, bins=n_bins, edgecolor='white',
                                 linewidth=0.6, color=BLUE, alpha=0.75)

# Colour significant vs not
for patch, left in zip(patches, bins[:-1]):
    right = left + (bins[1]-bins[0])
    # approximate: lift > 0 & p<0.05 — colour by sign
    patch.set_facecolor(GREEN if left > 0 else RED)
    patch.set_alpha(0.75)

ax.axvline(mean_lift, color='#333333', linewidth=2.2,
           label=f'Mean lift = {mean_lift:.1f}%')
ax.axvline(ci_low,  color='#333333', linewidth=1.4, linestyle='--',
           label=f'95% CI  [{ci_low:.1f}%, {ci_high:.1f}%]')
ax.axvline(ci_high, color='#333333', linewidth=1.4, linestyle='--')
ax.axvline(0, color=RED, linewidth=1.2, linestyle=':')

ax.set_xlabel('Simulated Lift (%)', fontsize=12)
ax.set_ylabel('Number of Simulations', fontsize=12)
ax.set_title(f'Distribution of Simulated Lift Across {N_SIMS:,} A/B Tests\n'
             f'n={n_required:,} per group | {pct_sig:.0f}% significant | '
             f'Mean lift={mean_lift:.1f}%', pad=14)

green_p = mpatches.Patch(color=GREEN, alpha=0.75, label='Positive lift')
red_p   = mpatches.Patch(color=RED,   alpha=0.75, label='Negative lift')
ax.legend(handles=[green_p, red_p] +
          [plt.Line2D([0],[0],color='#333333',lw=2.2,label=f'Mean={mean_lift:.1f}%'),
           plt.Line2D([0],[0],color='#333333',lw=1.4,linestyle='--',
                      label=f'95% CI [{ci_low:.1f}%,{ci_high:.1f}%]')],
          fontsize=10, framealpha=0.9)
sns.despine(ax=ax)
plt.tight_layout()
plt.savefig('outputs/ab1_lift_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print('\n  Saved: outputs/ab1_lift_distribution.png')

# ── VIZ 2: Power curve ─────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(sample_sizes, [p*100 for p in powers_mde], color=BLUE, linewidth=2.2,
        label=f'MDE = {MDE:.0%} absolute reduction')
ax.plot(sample_sizes, [p*100 for p in powers_trt], color=GREEN, linewidth=2.2,
        label=f'Treatment = 25% relative reduction ({BASELINE-TREAT_RATE:.2%} abs)')

ax.axhline(80, color='#333333', linestyle='--', linewidth=1.4,
           label='Target power = 80%')
ax.axvline(n_required, color=RED, linestyle='--', linewidth=1.4,
           label=f'Required n = {n_required:,}')

ax.fill_between(sample_sizes, [p*100 for p in powers_trt], 80,
                where=[p >= 0.80 for p in powers_trt],
                alpha=0.10, color=GREEN, label='Powered region (treatment)')

ax.set_xlabel('Sample Size per Group', fontsize=12)
ax.set_ylabel('Statistical Power (%)', fontsize=12)
ax.set_title(f'Power Curve — Sample Size vs Power\n'
             f'α={ALPHA} | Baseline={BASELINE:.2%} | Required n={n_required:,}', pad=14)
ax.set_xlim(200, 6000)
ax.set_ylim(0, 105)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}%'))
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x):,}'))
ax.legend(fontsize=10, framealpha=0.9)
sns.despine(ax=ax)
plt.tight_layout()
plt.savefig('outputs/ab2_power_curve.png', dpi=150, bbox_inches='tight')
plt.close()
print('  Saved: outputs/ab2_power_curve.png')

# ── VIZ 3: Significance rate vs effect size ───────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))

effect_pct = effect_sizes * 100
ax.plot(effect_pct, sig_rates, color=BLUE, linewidth=2.5,
        marker='o', markersize=8, markerfacecolor='white',
        markeredgewidth=2, markeredgecolor=BLUE)

for x, y in zip(effect_pct, sig_rates):
    ax.annotate(f'{y:.0f}%', (x, y), textcoords='offset points',
                xytext=(0, 10), ha='center', fontsize=9, fontweight='bold')

ax.axhline(80, color='#333333', linestyle='--', linewidth=1.4,
           label='80% power threshold')
ax.axvline(REL_LIFT*100, color=RED, linestyle='--', linewidth=1.4,
           label=f'Our treatment ({REL_LIFT:.0%} relative lift)')

ax.fill_between(effect_pct, sig_rates, 80,
                where=[s >= 80 for s in sig_rates],
                alpha=0.12, color=GREEN, label='Sufficiently powered')

ax.set_xlabel('Relative Churn Reduction (%)', fontsize=12)
ax.set_ylabel('% of Simulations Significant (p < 0.05)', fontsize=12)
ax.set_title(f'Detection Rate vs Effect Size\n'
             f'n={n_required:,} per group | α={ALPHA}', pad=14)
ax.set_xlim(0, 45)
ax.set_ylim(0, 110)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}%'))
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}%'))
ax.legend(fontsize=10, framealpha=0.9)
sns.despine(ax=ax)
plt.tight_layout()
plt.savefig('outputs/ab3_significance_vs_effect.png', dpi=150, bbox_inches='tight')
plt.close()
print('  Saved: outputs/ab3_significance_vs_effect.png')

# ── VIZ 4: P-value distribution ───────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Simulation Diagnostics', fontsize=14, fontweight='bold', y=1.01)

# P-value histogram
axes[0].hist(p_values, bins=40, color=BLUE, edgecolor='white',
             linewidth=0.6, alpha=0.80)
axes[0].axvline(ALPHA, color=RED, linewidth=2,
                label=f'α = {ALPHA}  ({pct_sig:.0f}% below)')
sig_area = (p_values < ALPHA).sum()
axes[0].set_xlabel('p-value', fontsize=11)
axes[0].set_ylabel('Count', fontsize=11)
axes[0].set_title(f'P-value Distribution\n{sig_area:,}/{N_SIMS:,} sims significant',
                  fontweight='bold')
axes[0].legend(fontsize=10, framealpha=0.9)
sns.despine(ax=axes[0])

# Churn rate distributions: control vs treatment
axes[1].hist(ctrl_rates, bins=40, alpha=0.65, color=RED,
             edgecolor='white', linewidth=0.6, label=f'Control  (mean={mean_ctrl:.2f}%)')
axes[1].hist(trt_rates,  bins=40, alpha=0.65, color=GREEN,
             edgecolor='white', linewidth=0.6, label=f'Treatment (mean={mean_trt:.2f}%)')
axes[1].axvline(mean_ctrl, color=RED,   linewidth=2, linestyle='--')
axes[1].axvline(mean_trt,  color=GREEN, linewidth=2, linestyle='--')
axes[1].set_xlabel('Observed Churn Rate (%)', fontsize=11)
axes[1].set_ylabel('Count', fontsize=11)
axes[1].set_title('Churn Rate Distributions\nControl vs Treatment',
                  fontweight='bold')
axes[1].legend(fontsize=10, framealpha=0.9)
sns.despine(ax=axes[1])

plt.tight_layout()
plt.savefig('outputs/ab4_diagnostics.png', dpi=150, bbox_inches='tight')
plt.close()
print('  Saved: outputs/ab4_diagnostics.png')

# ═══════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════════════════
print(f'\n{SEP}')
print('FINAL SUMMARY')
print(SEP)

print(f"""
  POWER ANALYSIS:
  ───────────────────────────────────────────────────────────────
  Baseline churn rate          : {BASELINE:.2%}
  Minimum detectable effect    : {MDE:.2%} absolute reduction
  Required n per group         : {n_required:,}
  Total customers needed       : {n_required*2:,}
  Theoretical power @ MDE      : {pwr_actual:.2%}
  Theoretical power @ treatment: {pwr_actual_treat:.2%}

  SIMULATION ({N_SIMS:,} runs):
  ───────────────────────────────────────────────────────────────
  Mean control churn rate      : {mean_ctrl:.2f}%
  Mean treatment churn rate    : {mean_trt:.2f}%
  Mean lift                    : {mean_lift:.1f}%
  95% CI on lift               : [{ci_low:.1f}%, {ci_high:.1f}%]
  % simulations significant    : {pct_sig:.1f}%  ← empirical power

  BUSINESS INTERPRETATION:
  ───────────────────────────────────────────────────────────────
  With n={n_required:,} customers per group, we will detect a 25% relative
  churn reduction approximately {pct_sig:.0f}% of the time at α={ALPHA}.

  If the intervention truly reduces churn from {BASELINE:.2%} to {TREAT_RATE:.2%}:
  • Expected churners saved per cohort : {int((BASELINE-TREAT_RATE)*n_required):,}
  • Revenue saved per test cohort       : ${int((BASELINE-TREAT_RATE)*n_required*5000):,}
    (at $5,000 retention value per customer)
""")
print(SEP)

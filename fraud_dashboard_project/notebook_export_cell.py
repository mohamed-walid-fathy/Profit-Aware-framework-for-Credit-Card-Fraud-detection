# ════════════════════════════════════════════════════════════════════════════
# DASHBOARD EXPORT — CAPACITY-BOUND SYSTEM (the final deployable model)
# Paste as the LAST cell, AFTER the capacity section (best_static_under_budget,
# the triage env, and K_FINAL must already be defined).
#
# Deploys: budget-re-tuned static policy + budget-pressure triage (k=K_FINAL),
# enforced by a binding review budget. The dashboard's reviewer/capacity numbers
# are derived from the SAME budget the model runs at. The bundle also snapshots the
# validation set + the FROZEN PPO policy, so the dashboard's reviewer slider can
# re-tune live (PPO stays frozen — only the static base + triage-k are re-derived).
# ════════════════════════════════════════════════════════════════════════════
import sys, os
PROJECT_ROOT = os.path.abspath("fraud_dashboard_project")   # folder containing fraudcore/  <-- edit if needed
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from fraudcore.dashboard_export import write_dashboard_outputs, save_bundle

# frozen headline PPO policy (works whether or not you applied the rename)
_policy_hl = cab_a4 if "cab_a4" in globals() else seq_a4

# ── staffing <-> budget (model and dashboard share these numbers) ────────────
REVIEWS_PER_DAY = 100                                    # REVIEWS_PER_EMPLOYEE_PER_DAY
_dt = raw_test_df["TransactionDT"].to_numpy()
TXNS_PER_DAY = len(raw_test_df) / ((_dt.max() - _dt.min()) / 86400)
N_REVIEWERS = 6                                          # initial staffing; the dashboard slider overrides this live
DEPLOY_BUDGET_FRAC = (N_REVIEWERS * REVIEWS_PER_DAY) / TXNS_PER_DAY

# ── the deployed capacity-bound policy (re-tuned from the FROZEN PPO policy) ──
OOB = "threshold"
_best_a, _ = best_static_under_budget(_policy_hl, DEPLOY_BUDGET_FRAC, split="val", oob=OOB)
_k = float(K_FINAL) if "K_FINAL" in globals() else 0.0

# kwargs for the file writer (build_dashboard_payload has a fixed signature, so val_*
# arrays are passed to save_bundle ONLY, not here)
_common = dict(
    scores=test_probs, labels=true_test_labels, amounts=test_amounts, meta_df=raw_test_df,
    policy_a4=_best_a,                                    # deployed (budget-re-tuned) base
    headline_policy_a4=_policy_hl,                        # frozen PPO -> "RL (amount-conditioned)" comparison row
    z_mean=Z_LOGA_MEAN, z_std=Z_LOGA_STD, min_t_low=MIN_T_LOW, max_t_high=MAX_T_HIGH,
    mode="capacity",
    budget_frac=DEPLOY_BUDGET_FRAC, triage_k=_k, gate="pressure", oob=OOB,
    txns_per_day=TXNS_PER_DAY,
    reviewers=N_REVIEWERS, reviews_per_reviewer_per_day=REVIEWS_PER_DAY,
    grid_thresholds=(gs_T_low, gs_T_high),
    single_thresholds=tuple(thr) if "thr" in globals() else None,
    bucketed_decisions=test_decisions if "test_decisions" in globals() else None,
    auc_val=val_auc_final if "val_auc_final" in globals() else None,
    auc_test=test_auc_final if "test_auc_final" in globals() else None,
)

# snapshot for live re-tuning (val set + frozen policy) and for regenerate_dashboard.py
save_bundle(os.path.join(PROJECT_ROOT, "dashboard_outputs", "deployment_bundle.npz"),
            val_scores=val_probs, val_labels=true_val_labels, val_amounts=val_amounts,
            **_common)
res = write_dashboard_outputs(os.path.join(PROJECT_ROOT, "dashboard_outputs"), **_common)

m = res["metrics"]
print(f"Deployed capacity-bound system  (budget {100*DEPLOY_BUDGET_FRAC:.1f}% ≈ {N_REVIEWERS} analysts × {REVIEWS_PER_DAY}/day,  triage k={_k:g})")
print(f"  review_rate = {100*m['review_rate']:.1f}%   profit_gain = {100*m['profit_gain']:.1f}%   total_profit = ${m['total_profit']:,.0f}")
print(f"  overload (force-accepted fraud + force-rejected legit) = ${m['overload_cost_total']:,.0f}")
print(f"  reviewers in rl_policy.json = {res['policy']['reviewers']}  (daily capacity {res['policy']['reviewers']*REVIEWS_PER_DAY:,} vs ~{TXNS_PER_DAY:,.0f} txns/day)")
print("  bundle includes val set + frozen PPO policy -> the dashboard reviewer slider can re-tune live.")

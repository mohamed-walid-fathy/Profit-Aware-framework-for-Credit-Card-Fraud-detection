"""Fast re-tuning of the capacity-bound system when staffing changes.

The PPO policy (`cab_a4`) is FROZEN. Changing the reviewer count only changes the review
budget, which requires re-deriving two cheap, val-tuned pieces — no gradient training:

    1. best_static_under_budget : a 9x6 grid sweep of base_low / base_high offsets on the
       frozen policy, scored under the budget on validation (replicates the notebook).
    2. tune_triage_k            : the budget-pressure triage coefficient, picked from a
       small grid on validation at this budget (replicates the K_FINAL search, per-budget).

Both are vectorised numpy rollouts (no autograd), so a full re-tune is sub-second — which
is what lets the dashboard's reviewer control update live.
"""
import numpy as np

from .costs import DEFAULT_COST_PARAMS
from .policy import capacity_constrained_decisions
from .metrics import profit_and_gain

# notebook grids (cells 66 / 79)
DEFAULT_DLO_GRID = np.linspace(-0.08, 0.24, 9)
DEFAULT_DHI_GRID = np.linspace(-0.20, 0.05, 6)
DEFAULT_K_GRID = [0, 4, 8, 12, 16, 20, 24, 28, 32]


def _score_under_budget(policy_a4, scores, labels, amounts, z_mean, z_std,
                        min_t_low, max_t_high, budget_frac, oob, triage_k, gate,
                        batch_size, cp):
    """Profit of `policy_a4` under the budget on this split (env-exact: cover_all=False)."""
    cap = capacity_constrained_decisions(
        policy_a4, scores, amounts, labels, z_mean, z_std, min_t_low, max_t_high,
        budget_frac=budget_frac, oob=oob, triage_k=triage_k, gate=gate,
        batch_size=batch_size, cp=cp, cover_all=False)
    cov = cap["covered"]
    return profit_and_gain(cap["decisions"][:cov], labels[:cov], amounts[:cov], cp)["model_profit"]


def best_static_under_budget(base_policy, scores, labels, amounts, z_mean, z_std,
                             min_t_low, max_t_high, *, budget_frac, oob="threshold",
                             batch_size=512, cp=DEFAULT_COST_PARAMS,
                             dlo_grid=None, dhi_grid=None):
    """Best open-loop static policy under the budget: 2-D sweep of base_low / base_high
    offsets on `base_policy` (the frozen PPO 4-vector), scored on `scores/labels/amounts`
    under budget enforcement. Tune on validation. Mirrors the notebook exactly."""
    dlo_grid = DEFAULT_DLO_GRID if dlo_grid is None else dlo_grid
    dhi_grid = DEFAULT_DHI_GRID if dhi_grid is None else dhi_grid
    base = np.asarray(base_policy, float)
    best, bestp = None, -np.inf
    for dlo in dlo_grid:
        for dhi in dhi_grid:
            a = base.copy(); a[0] += dlo; a[2] += dhi
            prof = _score_under_budget(a, scores, labels, amounts, z_mean, z_std,
                                       min_t_low, max_t_high, budget_frac, oob, 0.0, "off",
                                       batch_size, cp)
            if prof > bestp:
                bestp = prof; best = a
    return best, float(bestp)


def tune_triage_k(base_policy, scores, labels, amounts, z_mean, z_std,
                  min_t_low, max_t_high, *, budget_frac, oob="threshold",
                  batch_size=512, cp=DEFAULT_COST_PARAMS, k_grid=None, gate="pressure"):
    """Pick the budget-pressure triage coefficient on validation AT THIS budget (argmax
    profit over the grid). Per-budget re-tune — the piece that changes with staffing."""
    k_grid = DEFAULT_K_GRID if k_grid is None else k_grid
    bestk, bestv = 0.0, -np.inf
    for k in k_grid:
        v = _score_under_budget(base_policy, scores, labels, amounts, z_mean, z_std,
                                min_t_low, max_t_high, budget_frac, oob, float(k), gate,
                                batch_size, cp)
        if v > bestv:
            bestv, bestk = v, float(k)
    return bestk, float(bestv)


def budget_frac_from_reviewers(reviewers, reviews_per_reviewer_per_day, txns_per_day):
    """The single tie between staffing and the model's budget (capped at 1.0 = non-binding)."""
    return float(min((reviewers * reviews_per_reviewer_per_day) / txns_per_day, 1.0))


def deploy_capacity_system(reviewers, *, val_scores, val_labels, val_amounts,
                           headline_policy, z_mean, z_std, min_t_low, max_t_high,
                           reviews_per_reviewer_per_day=100, txns_per_day=None,
                           oob="threshold", batch_size=512, cp=DEFAULT_COST_PARAMS,
                           dlo_grid=None, dhi_grid=None, k_grid=None,
                           retune_k=True, fixed_k=0.0):
    """Re-tune the deployable capacity-bound policy for a given reviewer count.

    PPO is frozen; this only re-derives the budget-re-tuned static base (val) and the
    triage coefficient (val), both at the budget implied by `reviewers`. Returns the
    deployable params; feed `best_a` / `triage_k` / `budget_frac` to the test-side export.
    """
    if txns_per_day is None or txns_per_day <= 0:
        raise ValueError("deploy_capacity_system needs a positive txns_per_day")
    budget_frac = budget_frac_from_reviewers(reviewers, reviews_per_reviewer_per_day, txns_per_day)
    best_a, val_profit = best_static_under_budget(
        headline_policy, val_scores, val_labels, val_amounts, z_mean, z_std,
        min_t_low, max_t_high, budget_frac=budget_frac, oob=oob, batch_size=batch_size,
        cp=cp, dlo_grid=dlo_grid, dhi_grid=dhi_grid)
    if retune_k:
        triage_k, _ = tune_triage_k(
            best_a, val_scores, val_labels, val_amounts, z_mean, z_std,
            min_t_low, max_t_high, budget_frac=budget_frac, oob=oob, batch_size=batch_size,
            cp=cp, k_grid=k_grid)
    else:
        triage_k = float(fixed_k)
    return dict(reviewers=int(reviewers), budget_frac=budget_frac,
                best_a=np.asarray(best_a, float), triage_k=float(triage_k),
                val_profit=float(val_profit))

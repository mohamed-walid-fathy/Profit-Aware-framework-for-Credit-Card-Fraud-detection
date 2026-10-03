"""Decision policies: the amount-conditioned PPO policy and a global-threshold policy.

The threshold math is ported verbatim from the notebook's `amountcond_metrics`
(and the amount-aware floor from `compute_financial_metrics`) so per-transaction
decisions and thresholds are identical to what the thesis evaluates.
"""
import numpy as np

from .costs import DEFAULT_COST_PARAMS


def amount_conditioned_thresholds(a4, amounts, z_mean, z_std,
                                  min_t_low, max_t_high, cp=DEFAULT_COST_PARAMS):
    """Per-transaction (effective_T_low, T_high) for the 4-parameter policy.

    a4 = (base_low, slope_low, base_high, slope_high)
    z(A) is standardised log-amount, frozen on validation (Z_LOGA_MEAN/STD).
    Returns the *effective* lower threshold (after the rc/(flm*A) floor) and the
    upper threshold — exactly the two curves the dashboard plots.
    """
    pr, ltv, flm, rc, prev = cp['pr'], cp['ltv'], cp['flm'], cp['rc'], cp['prev']
    A = np.asarray(amounts, float)
    base_low, slope_low, base_high, slope_high = [float(v) for v in a4]
    logA = np.log1p(A)
    z = (logA - z_mean) / z_std
    T_low = base_low + slope_low * z
    T_high = base_high + slope_high * z
    T_low = np.clip(T_low, min_t_low, max_t_high - 0.02)
    T_high = np.clip(T_high, T_low + 0.01, max_t_high)
    eff_low = np.minimum(np.maximum(T_low, rc / (flm * A + 1e-6)), T_high - 0.01)
    return eff_low, T_high


def decisions_from_thresholds(scores, eff_low, t_high):
    """0=accept, 1=reject, 2=review from per-transaction thresholds."""
    p = np.asarray(scores, float)
    eff_low = np.asarray(eff_low, float)
    t_high = np.asarray(t_high, float)
    return np.where(p < eff_low, 0, np.where(p < t_high, 2, 1)).astype(int)


def amount_conditioned_decisions(a4, scores, amounts, z_mean, z_std,
                                 min_t_low, max_t_high, cp=DEFAULT_COST_PARAMS):
    """Convenience: returns (decisions, effective_T_low, T_high)."""
    eff_low, t_high = amount_conditioned_thresholds(
        a4, amounts, z_mean, z_std, min_t_low, max_t_high, cp)
    d = decisions_from_thresholds(scores, eff_low, t_high)
    return d, eff_low, t_high


def profit_optimal_binary_cutoff(cp=DEFAULT_COST_PARAMS):
    """The oob='threshold' fallback cutoff p* used when a review can't be granted:
    reject if p >= p*, else accept. p* = pr*(1+ltv) / (flm + pr*(1+ltv))."""
    pr, ltv, flm = cp['pr'], cp['ltv'], cp['flm']
    return pr * (1 + ltv) / (flm + pr * (1 + ltv) + 1e-12)


def capacity_constrained_decisions(policy_a4, scores, amounts, labels,
                                   z_mean, z_std, min_t_low, max_t_high, *,
                                   budget_frac, oob='threshold', triage_k=0.0,
                                   gate='off', batch_size=512,
                                   cp=DEFAULT_COST_PARAMS, cover_all=True):
    """Replicates CapacityConstrainedEnv / CapacityAblationEnv `.step` exactly.

    Processes the time-ordered stream in batches; a cumulative review budget is spent
    earliest-first, and once exhausted any further would-be review overflows to the
    `oob` fallback ('threshold' = profit-optimal binary cutoff, the notebook default).
    `gate='pressure'` with `triage_k>0` applies the budget-pressure triage: bump
    effective_T_low up for small-amount transactions when running behind on budget.

    Returns a dict of per-transaction arrays:
        decisions  : realized 0/1/2 after overflow
        desired    : 0/1/2 the policy wanted before overflow (post-triage-bump)
        eff_low    : effective lower threshold actually used (incl. triage bump)
        t_high     : upper threshold
        covered    : number of rows processed
        budget_left, total_budget
    """
    p = np.asarray(scores, float); A = np.asarray(amounts, float); y = np.asarray(labels, int)
    n = len(p)
    logA = np.log1p(A); z = (logA - z_mean) / z_std
    n_full = n // batch_size                                   # env's self.n_batches
    n_batches = ((n + batch_size - 1) // batch_size) if cover_all else n_full
    covered = n if cover_all else n_full * batch_size
    total_budget = max(1.0, budget_frac * covered)
    pstar = profit_optimal_binary_cutoff(cp)
    pr, ltv, flm, rc, prev = cp['pr'], cp['ltv'], cp['flm'], cp['rc'], cp['prev']
    base_low, slope_low, base_high, slope_high = [float(v) for v in policy_a4]

    d_real = np.full(n, -1, int); d_des = np.full(n, -1, int)
    eff_used = np.full(n, np.nan, float); thi = np.full(n, np.nan, float)
    budget_left = total_budget

    for b in range(n_batches):
        lo = b * batch_size; hi = min(lo + batch_size, n)
        pb = p[lo:hi]; Ab = A[lo:hi]; zb = z[lo:hi]
        T_low = base_low + slope_low * zb
        T_high = base_high + slope_high * zb
        T_low = np.clip(T_low, min_t_low, max_t_high - 0.02)
        T_high = np.clip(T_high, T_low + 0.01, max_t_high)
        eff = np.minimum(np.maximum(T_low, rc / (flm * Ab + 1e-6)), T_high - 0.01)
        if triage_k > 0 and gate != 'off':
            small = np.maximum(0.0, -zb)
            if gate == 'amount':
                bump = triage_k * small
            else:  # 'pressure' — frac uses the env's n_batches (= n // batch_size)
                frac = b / max(1, n_full)
                brem = budget_left / total_budget
                bump = triage_k * max(0.0, (1.0 - frac) - brem) * small
            eff = np.minimum(eff + bump, T_high - 0.01)
        dd = np.where(pb < eff, 0, np.where(pb < T_high, 2, 1)).astype(int)   # desired
        dr = dd.copy()
        rev_idx = np.where(dd == 2)[0]
        n_des = len(rev_idx)
        n_grant = int(min(n_des, max(0.0, budget_left)))
        if n_grant < n_des:
            den = rev_idx[n_grant:]
            if oob == 'accept':
                dr[den] = 0
            elif oob == 'reject':
                dr[den] = 1
            else:  # 'threshold'
                dr[den] = np.where(pb[den] >= pstar, 1, 0)
        budget_left -= n_grant
        d_des[lo:hi] = dd; d_real[lo:hi] = dr
        eff_used[lo:hi] = eff; thi[lo:hi] = T_high

    return dict(decisions=d_real, desired=d_des, eff_low=eff_used, t_high=thi,
                covered=int(covered), total_budget=float(total_budget),
                budget_left=float(budget_left))


def global_threshold_decisions(scores, amounts, t_low, t_high, cp=DEFAULT_COST_PARAMS):
    """Single global (T_low, T_high) with the same amount-aware floor used everywhere.

    Returns (decisions, effective_T_low, effective_T_high) where effective_T_high is a
    constant vector at t_high (kept as a vector so it can sit next to the amount-
    conditioned policy in the same table).
    """
    pr, ltv, flm, rc, prev = cp['pr'], cp['ltv'], cp['flm'], cp['rc'], cp['prev']
    A = np.asarray(amounts, float)
    p = np.asarray(scores, float)
    t_low = float(np.clip(t_low, 0.0, 1.0))
    t_high = float(np.clip(t_high, max(t_low, 0.0), 1.0))
    eff_low = np.minimum(np.maximum(t_low, rc / (flm * A + 1e-6)), t_high - 0.01)
    d = np.where(p < eff_low, 0, np.where(p < t_high, 2, 1)).astype(int)
    eff_high = np.full_like(A, t_high, dtype=float)
    return d, eff_low, eff_high

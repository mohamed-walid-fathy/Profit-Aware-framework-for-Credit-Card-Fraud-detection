"""Cost function and per-transaction cost decomposition.

`transaction_cost` is copied verbatim from the notebook (cell with the comment
"Single source of truth (B3)"). `cost_components` splits the realised cost into the
positive buckets the dashboard charts expect, plus protected exposure.
"""
import numpy as np

# Cost / reward parameters (Section 3 of the paper). Matches the notebook's COST_PARAMS.
DEFAULT_COST_PARAMS = dict(
    pr=0.02,    # profit rate on a legit transaction
    ltv=5,      # customer lifetime-value multiplier (cost of a false rejection)
    flm=3,      # fraud-loss multiplier (cost of a missed fraud)
    rc=3.5,     # manual review cost per transaction
    prev=1.0,   # reviewer accuracy (1.0 = reviewers are always correct)
)


def transaction_cost(d, y, A, cp=DEFAULT_COST_PARAMS):
    """Per-transaction cost (= negative profit). d: 0=accept, 1=reject, 2=review.

    Verbatim from the notebook so totals reconcile exactly:
        accept : fraud -> flm*A          legit -> -pr*A   (a gain)
        reject : fraud -> 0              legit ->  pr*A*ltv
        review : legit -> rc + pr*A*((1-prev)*ltv - prev)
                 fraud -> rc + (1-prev)*flm*A
    """
    pr, ltv, flm, rc, prev = cp['pr'], cp['ltv'], cp['flm'], cp['rc'], cp['prev']
    A = np.asarray(A, float); y = np.asarray(y, int); d = np.asarray(d, int)
    return np.where(
        d == 0, np.where(y == 1, flm * A, -pr * A),
        np.where(d == 1, np.where(y == 1, 0.0, pr * A * ltv),
                 np.where(y == 0, rc + pr * A * ((1 - prev) * ltv - prev),
                          rc + (1 - prev) * flm * A)))


def cost_components(d, y, A, cp=DEFAULT_COST_PARAMS):
    """Decompose into the dashboard's positive cost buckets and protected exposure.

    These are the *gross* operational/fraud costs the cost-breakdown chart shows
    (they deliberately exclude the -pr*A gain on accepted-legit traffic; that gain
    is reflected only in the net profit / financial-outcome view).

        missed_fraud_cost    = flm * A      on fraud that was ACCEPTED
        false_rejection_cost = pr*A*ltv     on legit that was REJECTED
        manual_review_cost   = rc           on anything REVIEWED
        overload_cost        = 0            (set by the caller for capacity-constrained runs)
        protected_exposure   = A            on actual fraud NOT accepted (caught by review/reject)
    """
    pr, ltv, flm, rc, prev = cp['pr'], cp['ltv'], cp['flm'], cp['rc'], cp['prev']
    A = np.asarray(A, float); y = np.asarray(y, int); d = np.asarray(d, int)
    missed = np.where((d == 0) & (y == 1), flm * A, 0.0)
    false_rej = np.where((d == 1) & (y == 0), pr * A * ltv, 0.0)
    review = np.where(d == 2, float(rc), 0.0)
    overload = np.zeros_like(A, dtype=float)
    protected = np.where((y == 1) & (d != 0), A, 0.0)
    return dict(
        missed_fraud_cost=missed,
        false_rejection_cost=false_rej,
        manual_review_cost=review,
        overload_cost=overload,
        protected_exposure=protected,
    )


def capacity_cost_components(d_real, d_desired, y, A, cp=DEFAULT_COST_PARAMS):
    """Cost decomposition for the capacity-constrained system, with an honest overload bucket.

    `d_desired` is what the policy wanted (pre-overflow); `d_real` is what happened after
    the review budget was enforced. A transaction is "overflowed" when it was desired for
    review but couldn't be granted one (forced to the oob fallback). The cost of those
    forced conversions goes into `overload_cost` rather than missed_fraud / false_rejection,
    so the overload bar shows exactly the damage caused by being under capacity:

        forced fraud  -> accept : overload += flm*A   (would have been caught by review)
        forced legit  -> reject : overload += pr*A*ltv (would have been approved by review)
        forced legit  -> accept : 0  (no damage; review wasn't needed)
        forced fraud  -> reject : 0  (still caught)

    The four positive buckets sum to the same gross cost as `cost_components`; only the
    attribution of forced transactions differs. Net profit (with the -pr*A legit gains)
    is computed separately in metrics and is unaffected.
    """
    pr, ltv, flm, rc, prev = cp['pr'], cp['ltv'], cp['flm'], cp['rc'], cp['prev']
    A = np.asarray(A, float); y = np.asarray(y, int)
    d_real = np.asarray(d_real, int); d_desired = np.asarray(d_desired, int)
    forced = (d_desired == 2) & (d_real != 2)

    missed = np.where((d_real == 0) & (y == 1) & (~forced), flm * A, 0.0)
    false_rej = np.where((d_real == 1) & (y == 0) & (~forced), pr * A * ltv, 0.0)
    review = np.where(d_real == 2, float(rc), 0.0)
    overload = (np.where(forced & (d_real == 0) & (y == 1), flm * A, 0.0)
                + np.where(forced & (d_real == 1) & (y == 0), pr * A * ltv, 0.0))
    protected = np.where((y == 1) & (d_real != 0), A, 0.0)
    return dict(
        missed_fraud_cost=missed,
        false_rejection_cost=false_rej,
        manual_review_cost=review,
        overload_cost=overload,
        protected_exposure=protected,
    )

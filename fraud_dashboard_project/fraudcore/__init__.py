"""fraudcore — single source of truth for the cost model, policy decisions, and metrics
shared by the thesis notebook and the Streamlit dashboard."""
from .costs import (
    DEFAULT_COST_PARAMS, transaction_cost, cost_components, capacity_cost_components,
)
from .policy import (
    amount_conditioned_thresholds,
    amount_conditioned_decisions,
    capacity_constrained_decisions,
    profit_optimal_binary_cutoff,
    global_threshold_decisions,
    decisions_from_thresholds,
)
from .metrics import profit_and_gain, confusion_counts, classification_rates
from .tuning import (
    best_static_under_budget, tune_triage_k, deploy_capacity_system,
    budget_frac_from_reviewers,
)
from .dashboard_export import (
    build_dashboard_payload, write_dashboard_outputs, live_dashboard_payload,
    build_transactions_frame, build_policy_comparison, save_bundle, load_bundle,
)

__all__ = [
    "DEFAULT_COST_PARAMS", "transaction_cost", "cost_components", "capacity_cost_components",
    "amount_conditioned_thresholds", "amount_conditioned_decisions",
    "capacity_constrained_decisions", "profit_optimal_binary_cutoff",
    "global_threshold_decisions", "decisions_from_thresholds",
    "profit_and_gain", "confusion_counts", "classification_rates",
    "best_static_under_budget", "tune_triage_k", "deploy_capacity_system",
    "budget_frac_from_reviewers",
    "build_dashboard_payload", "write_dashboard_outputs", "live_dashboard_payload",
    "build_transactions_frame", "build_policy_comparison", "save_bundle", "load_bundle",
]

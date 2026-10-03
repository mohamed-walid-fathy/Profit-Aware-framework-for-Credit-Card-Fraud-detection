"""Write the four files the Streamlit dashboard reads, from trained artifacts.

The dashboard is a pure consumer: it reads
    dashboard_outputs/dashboard_transactions.csv
    dashboard_outputs/dashboard_metrics.json
    dashboard_outputs/rl_policy.json
    dashboard_outputs/policy_comparison.csv   (optional)
This module is the producer. Call `write_dashboard_outputs(...)` with the variables
that are already in scope at the end of the notebook (scores, labels, amounts, the
policy params, the frozen constants, the raw test frame). Nothing here re-trains.

`save_bundle` / `load_bundle` let you snapshot those inputs so the dashboard files can
be regenerated later (e.g. after relabelling a column) without rerunning the pipeline.
"""
import json
import os

import numpy as np
import pandas as pd

from .costs import DEFAULT_COST_PARAMS, cost_components, capacity_cost_components
from .policy import (
    amount_conditioned_decisions,
    capacity_constrained_decisions,
    global_threshold_decisions,
)
from .metrics import profit_and_gain, confusion_counts, classification_rates

_ACTION = {0: "accept", 1: "reject", 2: "review"}
_FINAL = {"accept": "approved", "review": "pending_review", "reject": "blocked"}


def _json_safe(obj):
    """Recursively coerce numpy scalars/arrays (and tuples) into JSON-serialisable types."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_json_safe(v) for v in obj.tolist()]
    if isinstance(obj, np.generic):       # np.float32/64, np.int64, np.bool_, ...
        return obj.item()
    return obj


def _txns_per_day_from_meta(meta_df, n):
    """Replicate the notebook's TXNS_PER_DAY = n / (span_seconds / 86400)."""
    if meta_df is not None and "TransactionDT" in meta_df.columns:
        dt = pd.to_numeric(meta_df["TransactionDT"], errors="coerce").to_numpy()[:n]
        span = float(np.nanmax(dt) - np.nanmin(dt))
        if span > 0:
            return n / (span / 86400.0)
    return None


# --------------------------------------------------------------------------- #
# transactions table
# --------------------------------------------------------------------------- #
def _meta_columns(meta_df, n):
    """Pull TransactionID / TransactionDT / card1 from the raw frame if present."""
    out = {}
    md = None
    if meta_df is not None:
        md = meta_df.reset_index(drop=True)
        md = md.iloc[:n] if len(md) >= n else md

    def col(name, default):
        if md is not None and name in md.columns and len(md) == n:
            return md[name].to_numpy()
        return default

    out["TransactionID"] = col("TransactionID", np.arange(n))
    out["TransactionDT"] = col("TransactionDT", np.zeros(n))
    out["card1"] = col("card1", np.full(n, -1))
    return out


def build_transactions_frame(scores, labels, amounts, decisions,
                             eff_low, t_high, meta_df, cp=DEFAULT_COST_PARAMS,
                             overload_cost=None, components=None):
    n = len(scores)
    comp = components if components is not None else cost_components(decisions, labels, amounts, cp)
    if overload_cost is not None:
        comp = dict(comp)
        comp["overload_cost"] = np.asarray(overload_cost, float)
    txn_cost = (comp["missed_fraud_cost"] + comp["false_rejection_cost"]
                + comp["manual_review_cost"] + comp["overload_cost"])
    meta = _meta_columns(meta_df, n)
    actions = np.array([_ACTION[int(v)] for v in decisions])
    tx = pd.DataFrame({
        "TransactionID": meta["TransactionID"],
        "TransactionDT": meta["TransactionDT"],
        "TransactionAmt": np.asarray(amounts, float),
        "isFraud": np.asarray(labels, int),
        "card1": meta["card1"],
        "catboost_risk_score": np.asarray(scores, float),
        "rl_action": actions,
        "effective_T_low": np.asarray(eff_low, float),
        "effective_T_high": np.asarray(t_high, float),
        "missed_fraud_cost": comp["missed_fraud_cost"],
        "false_rejection_cost": comp["false_rejection_cost"],
        "manual_review_cost": comp["manual_review_cost"],
        "overload_cost": comp["overload_cost"],
        "transaction_cost": txn_cost,
        "protected_exposure": comp["protected_exposure"],
    })
    tx["final_status"] = tx["rl_action"].map(_FINAL)
    tx["human_review_decision"] = np.where(
        tx["final_status"] == "pending_review", "not_reviewed_yet", "")
    return tx, comp, txn_cost


# --------------------------------------------------------------------------- #
# policy comparison
# --------------------------------------------------------------------------- #
def _comparison_row(name, d, scores, y, A, cp, t_low=np.nan, t_high=np.nan, components=None):
    comp = components if components is not None else cost_components(d, y, A, cp)
    gross = float((comp["missed_fraud_cost"] + comp["false_rejection_cost"]
                   + comp["manual_review_cost"] + comp["overload_cost"]).sum())
    pg = profit_and_gain(d, y, A, cp)
    return {
        "policy_name": name,
        "T_low": t_low,
        "T_high": t_high,
        "review_rate": float((np.asarray(d) == 2).mean()),
        "total_cost": gross,
        "profit_gain": pg["profit_gain"],
        "protected_exposure": float(comp["protected_exposure"].sum()),
    }


def build_policy_comparison(scores, y, A, cp, *, grid_thresholds, single_thresholds,
                            bucketed_decisions, amountcond, comparison_policies=None):
    """Return a tidy policy-comparison DataFrame (or None).

    `comparison_policies` is a list of specs; each is one of:
        (name, "global",     (t_low, t_high))
        (name, "decisions",  decision_vector)
        (name, "amountcond", (a4, z_mean, z_std, min_t_low, max_t_high))
        (name, "capacity",   (a4_static, z_mean, z_std, min_t_low, max_t_high,
                              budget_frac, triage_k, gate, oob, batch_size))
    If None, a sensible default mirroring the thesis stages is built from whatever
    of grid/bucketed/amountcond is available. `amountcond` should be the FROZEN headline
    policy (cab_a4), so the "RL (amount-conditioned)" row is the real Stage-2 policy.
    """
    if comparison_policies is None:
        comparison_policies = []
        if grid_thresholds is not None:
            comparison_policies.append(("Grid search (global)", "global", grid_thresholds))
        if bucketed_decisions is not None:
            comparison_policies.append(("Amount-bucketed grid", "decisions",
                                        np.asarray(bucketed_decisions)))
        comparison_policies.append(("RL (amount-conditioned)", "amountcond", amountcond))

    rows = []
    for spec in comparison_policies:
        name, kind, payload = spec[0], spec[1], spec[2]
        if kind == "global":
            tl, th = payload
            d, _, _ = global_threshold_decisions(scores, A, tl, th, cp)
            rows.append(_comparison_row(name, d, scores, y, A, cp,
                                        t_low=float(tl), t_high=float(th)))
        elif kind == "decisions":
            d = np.asarray(payload, int)
            rows.append(_comparison_row(name, d, scores, y, A, cp))
        elif kind == "amountcond":
            a4, zm, zs, mtl, mth = payload
            d, _, _ = amount_conditioned_decisions(a4, scores, A, zm, zs, mtl, mth, cp)
            rows.append(_comparison_row(name, d, scores, y, A, cp,
                                        t_low=float(a4[0]), t_high=float(a4[2])))
        elif kind == "capacity":
            a4, zm, zs, mtl, mth, bf, tk, gate, oob, bs = payload
            cap = capacity_constrained_decisions(a4, scores, A, y, zm, zs, mtl, mth,
                                                 budget_frac=bf, oob=oob, triage_k=tk,
                                                 gate=gate, batch_size=bs, cover_all=True)
            dd = cap["decisions"]
            comp = capacity_cost_components(dd, cap["desired"], y, A, cp)
            rows.append(_comparison_row(name, dd, scores, y, A, cp,
                                        t_low=float(a4[0]), t_high=float(a4[2]), components=comp))
        else:
            raise ValueError(f"unknown comparison policy kind: {kind!r}")
    return pd.DataFrame(rows) if rows else None


# --------------------------------------------------------------------------- #
# main entry point
# --------------------------------------------------------------------------- #
def build_dashboard_payload(*, scores, labels, amounts, meta_df,
                            policy_a4, z_mean, z_std, min_t_low, max_t_high,
                            mode="unconstrained",
                            budget_frac=None, triage_k=0.0, gate="off",
                            oob="threshold", batch_size=512, txns_per_day=None,
                            grid_thresholds=None, single_thresholds=None,
                            bucketed_decisions=None,
                            auc_val=None, auc_test=None,
                            reviewers=None, reviews_per_reviewer_per_day=100,
                            cost_params=DEFAULT_COST_PARAMS,
                            comparison_policies=None,
                            headline_policy_a4=None,
                            overload_cost=None,
                            score_is_hybrid=True):
    """Compute the dashboard payload in memory (no file writes). Returns a dict with
    keys: transactions, metrics, policy, comparison. `write_dashboard_outputs` writes it
    to disk; the live dashboard path calls this directly after re-tuning.

    `policy_a4` is the DEPLOYED policy (in capacity mode, the budget-re-tuned static base).
    `headline_policy_a4` is the FROZEN PPO policy (cab_a4) used for the "RL (amount-conditioned)"
    comparison row; defaults to `policy_a4` (correct for unconstrained mode).
    """
    scores = np.asarray(scores, float)
    y = np.asarray(labels, int)
    A = np.asarray(amounts, float)
    n = len(scores)
    cp = cost_params
    if headline_policy_a4 is None:
        headline_policy_a4 = policy_a4

    if mode == "capacity":
        if txns_per_day is None:
            txns_per_day = _txns_per_day_from_meta(meta_df, n)
        rpd = int(reviews_per_reviewer_per_day)
        if reviewers is not None and txns_per_day:
            budget_frac = (reviewers * rpd) / txns_per_day          # reviewers is authoritative
        elif budget_frac is not None and txns_per_day:
            reviewers = int(round(budget_frac * txns_per_day / rpd))
        if budget_frac is None:
            raise ValueError("capacity mode needs budget_frac or (reviewers + a TransactionDT span)")
        budget_frac = float(min(budget_frac, 1.0))   # capacity >= volume is non-binding

        cap = capacity_constrained_decisions(
            policy_a4, scores, A, y, z_mean, z_std, min_t_low, max_t_high,
            budget_frac=budget_frac, oob=oob, triage_k=triage_k, gate=gate,
            batch_size=batch_size, cp=cp, cover_all=True)
        d = cap["decisions"]
        eff_low, t_high = cap["eff_low"], cap["t_high"]
        comp = capacity_cost_components(d, cap["desired"], y, A, cp)
        policy_type = "capacity_constrained_static" + ("_triage" if (triage_k and gate != "off") else "")
    else:
        d, eff_low, t_high = amount_conditioned_decisions(
            policy_a4, scores, A, z_mean, z_std, min_t_low, max_t_high, cp)
        comp = cost_components(d, y, A, cp)
        if overload_cost is not None:
            comp = dict(comp); comp["overload_cost"] = np.asarray(overload_cost, float)
        if reviewers is None:
            reviewers = 8
        policy_type = "amount_conditioned_ppo"

    tx, comp, txn_cost = build_transactions_frame(
        scores, y, A, d, eff_low, t_high, meta_df, cp, components=comp)

    pg = profit_and_gain(d, y, A, cp)
    cm = confusion_counts(d, y)
    rates = classification_rates(cm)
    review_count = int((d == 2).sum())
    metrics = {
        "total_transactions": int(n),
        "accepted_count": int((d == 0).sum()),
        "review_count": review_count,
        "rejected_count": int((d == 1).sum()),
        "review_rate": review_count / max(n, 1),
        "auc_roc_val": None if auc_val is None else float(auc_val),
        "auc_roc_test": None if auc_test is None else float(auc_test),
        "precision": rates["precision"],
        "recall": rates["recall"],
        "false_positive_rate": rates["false_positive_rate"],
        "total_cost": float(txn_cost.sum()),
        "total_profit": pg["model_profit"],
        "profit_gain": pg["profit_gain"],
        "protected_exposure": float(comp["protected_exposure"].sum()),
        "missed_fraud_cost_total": float(comp["missed_fraud_cost"].sum()),
        "false_rejection_cost_total": float(comp["false_rejection_cost"].sum()),
        "manual_review_cost_total": float(comp["manual_review_cost"].sum()),
        "overload_cost_total": float(comp["overload_cost"].sum()),
        "tp": cm["tp"], "fp": cm["fp"], "tn": cm["tn"], "fn": cm["fn"],
        "no_fraud_value": pg["no_fraud_profit"],
        "model_policy_value": pg["model_profit"],
        "oracle_value": pg["oracle_profit"],
        "dummy_data": False,
    }
    metrics = {k: v for k, v in metrics.items() if v is not None}

    base_low, slope_low, base_high, slope_high = [float(v) for v in policy_a4]
    policy = {
        "policy_type": policy_type,
        "base_low": base_low, "slope_low": slope_low,
        "base_high": base_high, "slope_high": slope_high,
        "z_log_amount_mean": float(z_mean), "z_log_amount_std": float(z_std),
        "min_t_low": float(min_t_low), "max_t_high": float(max_t_high),
        "profit_rate": cp["pr"], "ltv": cp["ltv"],
        "fraud_loss_multiplier": cp["flm"], "review_cost": cp["rc"],
        "reviewer_accuracy": cp["prev"],
        "reviewers": int(reviewers),
        "reviews_per_reviewer_per_day": int(reviews_per_reviewer_per_day),
        "score_is_hybrid": bool(score_is_hybrid),
        "dummy_data": False,
    }
    if mode == "capacity":
        policy.update({
            "review_budget_frac": float(budget_frac),
            "triage_k": float(triage_k),
            "triage_gate": gate,
            "overflow_rule": oob,
            "txns_per_day": None if txns_per_day is None else float(txns_per_day),
        })

    # policy comparison: RL row uses the FROZEN headline policy; capacity mode also shows
    # the deployed (capacity-bound) operating point so the constraint's cost is explicit.
    if comparison_policies is None:
        comparison_policies = []
        if grid_thresholds is not None:
            comparison_policies.append(("Grid search (global)", "global", grid_thresholds))
        if bucketed_decisions is not None:
            comparison_policies.append(("Amount-bucketed grid", "decisions", np.asarray(bucketed_decisions)))
        comparison_policies.append(
            ("RL (amount-conditioned)", "amountcond",
             (headline_policy_a4, z_mean, z_std, min_t_low, max_t_high)))
        if mode == "capacity":
            comparison_policies.append(
                (f"Deployed (capacity, {int(reviewers)} reviewers)", "capacity",
                 (policy_a4, z_mean, z_std, min_t_low, max_t_high,
                  budget_frac, triage_k, gate, oob, batch_size)))
    pc = build_policy_comparison(
        scores, y, A, cp,
        grid_thresholds=grid_thresholds, single_thresholds=single_thresholds,
        bucketed_decisions=bucketed_decisions,
        amountcond=(headline_policy_a4, z_mean, z_std, min_t_low, max_t_high),
        comparison_policies=comparison_policies)

    return {"transactions": tx, "metrics": metrics, "policy": policy, "comparison": pc}


def write_dashboard_outputs(out_dir, **kwargs):
    """Compute the payload and write the four dashboard files into `out_dir`."""
    os.makedirs(out_dir, exist_ok=True)
    payload = build_dashboard_payload(**kwargs)
    payload["transactions"].to_csv(os.path.join(out_dir, "dashboard_transactions.csv"), index=False)
    with open(os.path.join(out_dir, "dashboard_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(_json_safe(payload["metrics"]), f, indent=2)
    with open(os.path.join(out_dir, "rl_policy.json"), "w", encoding="utf-8") as f:
        json.dump(_json_safe(payload["policy"]), f, indent=2)
    if payload["comparison"] is not None:
        payload["comparison"].to_csv(os.path.join(out_dir, "policy_comparison.csv"), index=False)
    return {**payload, "out_dir": out_dir}


def live_dashboard_payload(reviewers, *, val_scores, val_labels, val_amounts,
                           test_scores, test_labels, test_amounts, meta_df,
                           headline_policy_a4, z_mean, z_std, min_t_low, max_t_high,
                           reviews_per_reviewer_per_day=100, txns_per_day=None,
                           oob="threshold", batch_size=512, cost_params=DEFAULT_COST_PARAMS,
                           grid_thresholds=None, single_thresholds=None, bucketed_decisions=None,
                           auc_val=None, auc_test=None, score_is_hybrid=True,
                           retune_k=True, fixed_k=0.0):
    """Re-tune the capacity-bound system for `reviewers` (PPO frozen; only the static base
    and triage-k are re-derived on validation), then build the dashboard payload on test.
    This is what the dashboard's reviewer control calls."""
    from .tuning import deploy_capacity_system
    if txns_per_day is None:
        txns_per_day = _txns_per_day_from_meta(meta_df, len(test_scores))
    dep = deploy_capacity_system(
        reviewers, val_scores=val_scores, val_labels=val_labels, val_amounts=val_amounts,
        headline_policy=headline_policy_a4, z_mean=z_mean, z_std=z_std,
        min_t_low=min_t_low, max_t_high=max_t_high,
        reviews_per_reviewer_per_day=reviews_per_reviewer_per_day, txns_per_day=txns_per_day,
        oob=oob, batch_size=batch_size, cp=cost_params, retune_k=retune_k, fixed_k=fixed_k)
    payload = build_dashboard_payload(
        scores=test_scores, labels=test_labels, amounts=test_amounts, meta_df=meta_df,
        policy_a4=dep["best_a"], headline_policy_a4=headline_policy_a4,
        z_mean=z_mean, z_std=z_std, min_t_low=min_t_low, max_t_high=max_t_high,
        mode="capacity", budget_frac=dep["budget_frac"], triage_k=dep["triage_k"],
        gate="pressure", oob=oob, batch_size=batch_size, txns_per_day=txns_per_day,
        grid_thresholds=grid_thresholds, single_thresholds=single_thresholds,
        bucketed_decisions=bucketed_decisions, auc_val=auc_val, auc_test=auc_test,
        reviewers=reviewers, reviews_per_reviewer_per_day=reviews_per_reviewer_per_day,
        cost_params=cost_params, score_is_hybrid=score_is_hybrid)
    payload["deployment"] = dep
    return payload


# --------------------------------------------------------------------------- #
# bundle save/load (regenerate the dashboard files without rerunning training)
# --------------------------------------------------------------------------- #
def save_bundle(path, *, scores, labels, amounts, meta_df, policy_a4,
                z_mean, z_std, min_t_low, max_t_high,
                mode="unconstrained", budget_frac=None, triage_k=0.0, gate="off",
                oob="threshold", batch_size=512, txns_per_day=None,
                grid_thresholds=None, single_thresholds=None, bucketed_decisions=None,
                auc_val=None, auc_test=None,
                reviewers=None, reviews_per_reviewer_per_day=100,
                cost_params=DEFAULT_COST_PARAMS,
                val_scores=None, val_labels=None, val_amounts=None,
                headline_policy_a4=None, **_ignored):
    """Snapshot the export inputs to a .npz so dashboard files can be rebuilt later.

    Accepts the same kwargs as write_dashboard_outputs (extra kwargs are ignored), so you
    can splat the same dict into both.
    """
    if not path.endswith(".npz"):
        path = path + ".npz"
    meta_cols = {}
    if meta_df is not None:
        for c in ("TransactionID", "TransactionDT", "card1"):
            if c in meta_df.columns:
                meta_cols[f"meta_{c}"] = meta_df[c].to_numpy()
    scalars = _json_safe(dict(
        mode=mode, budget_frac=budget_frac, triage_k=triage_k, gate=gate, oob=oob,
        batch_size=batch_size, txns_per_day=txns_per_day,
        z_mean=z_mean, z_std=z_std, min_t_low=min_t_low, max_t_high=max_t_high,
        grid_thresholds=grid_thresholds, single_thresholds=single_thresholds,
        auc_val=auc_val, auc_test=auc_test, reviewers=reviewers,
        reviews_per_reviewer_per_day=reviews_per_reviewer_per_day,
        cost_params=cost_params))
    arrays = dict(scores=np.asarray(scores, float), labels=np.asarray(labels, int),
                  amounts=np.asarray(amounts, float), policy_a4=np.asarray(policy_a4, float),
                  _scalars_json=np.array(json.dumps(scalars)))
    if bucketed_decisions is not None:
        arrays["bucketed_decisions"] = np.asarray(bucketed_decisions, int)
    # the live reviewer control re-tunes on validation, so snapshot val data + frozen policy
    if val_scores is not None:
        arrays["val_scores"] = np.asarray(val_scores, float)
        arrays["val_labels"] = np.asarray(val_labels, int)
        arrays["val_amounts"] = np.asarray(val_amounts, float)
    if headline_policy_a4 is not None:
        arrays["headline_policy_a4"] = np.asarray(headline_policy_a4, float)
    arrays.update(meta_cols)
    np.savez(path, **arrays)
    return path


def load_bundle(path):
    """Load a saved bundle into the kwargs write_dashboard_outputs expects."""
    if not path.endswith(".npz"):
        path = path + ".npz"
    z = np.load(path, allow_pickle=True)
    s = json.loads(str(z["_scalars_json"]))
    meta = {c.replace("meta_", ""): z[c] for c in z.files if c.startswith("meta_")}
    meta_df = pd.DataFrame(meta) if meta else None
    kwargs = dict(
        scores=z["scores"], labels=z["labels"], amounts=z["amounts"],
        meta_df=meta_df, policy_a4=z["policy_a4"],
        z_mean=s["z_mean"], z_std=s["z_std"],
        min_t_low=s["min_t_low"], max_t_high=s["max_t_high"],
        mode=s.get("mode", "unconstrained"),
        budget_frac=s.get("budget_frac"), triage_k=s.get("triage_k", 0.0),
        gate=s.get("gate", "off"), oob=s.get("oob", "threshold"),
        batch_size=s.get("batch_size", 512), txns_per_day=s.get("txns_per_day"),
        grid_thresholds=tuple(s["grid_thresholds"]) if s["grid_thresholds"] else None,
        single_thresholds=tuple(s["single_thresholds"]) if s["single_thresholds"] else None,
        bucketed_decisions=z["bucketed_decisions"] if "bucketed_decisions" in z.files else None,
        auc_val=s["auc_val"], auc_test=s["auc_test"],
        reviewers=s["reviewers"],
        reviews_per_reviewer_per_day=s["reviews_per_reviewer_per_day"],
        cost_params=s["cost_params"],
    )
    for k in ("val_scores", "val_labels", "val_amounts", "headline_policy_a4"):
        if k in z.files:
            kwargs[k] = z[k]
    return kwargs

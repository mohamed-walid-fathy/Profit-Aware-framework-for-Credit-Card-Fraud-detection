# Fraud dashboard + notebook integration

The Streamlit dashboard and the research notebook are joined by a **file contract**, not
by importing the notebook. The dashboard reads four files; the notebook produces them
through a small shared library. The dashboard UI is unchanged.

```
fraud_dashboard_project/
├── dashboard.py                # the Streamlit app — byte-identical to your upload
├── notebook_export_cell.py     # paste this as the LAST cell of the notebook
├── regenerate_dashboard.py     # rebuild the 4 files from a snapshot, no retraining
├── requirements.txt
├── fraudcore/                  # shared math — the single source of truth
│   ├── costs.py                #   COST_PARAMS, transaction_cost, cost decomposition
│   ├── policy.py               #   amount-conditioned + global-threshold decisions
│   ├── metrics.py              #   profit / profit-gain, confusion counts, rates
│   └── dashboard_export.py     #   write_dashboard_outputs(...), policy comparison, bundle I/O
└── dashboard_outputs/          # generated: the 4 files the dashboard reads
    ├── dashboard_transactions.csv
    ├── dashboard_metrics.json
    ├── rl_policy.json
    └── policy_comparison.csv
```

## Why split it this way (the short answer to "divide into files, right?")

Yes — but the useful split is **not** chopping the notebook into modules. It is:

1. **`fraudcore/`** holds the cost function, the threshold policy, and the metrics —
   ported verbatim from the notebook. The export uses these, so the dashboard can never
   show numbers that disagree with the thesis. If a formula changes, it changes in one
   place. (The notebook can also `from fraudcore...` to drop its inline copies, but it
   doesn't have to.)
2. **`dashboard.py`** stays a pure consumer of files — it already worked that way, so it
   needs no changes and keeps looking exactly the same.
3. **`dashboard_export.py`** is the bridge: it turns trained artifacts into the four files.

The boundary is the export contract, which is the part that was actually missing.

## How to run

1. Train as usual in the notebook (or load checkpoints).
2. Paste `notebook_export_cell.py` as the last cell and run it. Set `PROJECT_ROOT` to this
   folder. It writes the four files into `dashboard_outputs/`.
3. Launch the dashboard:
   ```
   pip install -r requirements.txt
   streamlit run dashboard.py
   ```
   It defaults to reading `./dashboard_outputs`.

To refresh the files later **without** rerunning the notebook (the export cell also saved a
`deployment_bundle.npz`):
```
python regenerate_dashboard.py --reviewers 8 --reviews-per-day 100
```

## Data contract (what the export writes / the dashboard reads)

- **dashboard_transactions.csv** — one row per transaction: `TransactionID`,
  `TransactionDT`, `TransactionAmt`, `isFraud`, `card1`, `catboost_risk_score`,
  `rl_action` (accept/review/reject), `effective_T_low`, `effective_T_high`, the four cost
  components, `transaction_cost`, `protected_exposure`, `final_status`.
- **dashboard_metrics.json** — counts, AUCs, precision/recall/FPR, `total_cost` (gross),
  `profit_gain`, `protected_exposure`, the four `*_cost_total`s, `tp/fp/tn/fn`, and the
  net `no_fraud_value` / `model_policy_value` / `oracle_value`.
- **rl_policy.json** — the four policy parameters, the frozen `z` constants and bounds, the
  cost parameters, and `reviewers` / `reviews_per_reviewer_per_day`.
- **policy_comparison.csv** — one row per policy (default: grid search, amount-bucketed
  grid, RL amount-conditioned) with `T_low/T_high/review_rate/total_cost/profit_gain/
  protected_exposure`.

## Three things worth knowing (so the numbers read correctly)

- **`catboost_risk_score` is the hybrid score.** The deployed score is the AE+CatBoost
  blend (α≈0.95), so it is ~CatBoost but not identical. The column keeps the dashboard's
  name; `rl_policy.json.score_is_hybrid = true` records the truth.
- **`total_cost` is gross; the financial chart is net.** The cost-breakdown chart and the
  `total_cost` KPI sum the positive buckets (missed fraud + false rejection + review +
  overload). The financial-outcome chart uses the *true net* profit
  (`model_policy_value`), which also credits the `−pr·A` gain on accepted-legit traffic.
  Both are correct; they are different lenses. `profit_gain` is computed from the net
  profit and matches the notebook exactly.
- **Capacity utilisation.** The dashboard computes `review_count / (reviewers × reviews/
  day)`. `review_count` is over the whole ~30.8-day test span while the denominator is a
  *daily* capacity, so read that KPI as "reviews over the test period vs one day of
  capacity," or scale `reviewers` if you want a like-for-like ratio. The per-day picture
  is the "Daily Review Load" chart.

## Deploying the capacity-bound system (the final model)

`write_dashboard_outputs` has two deployment modes, selected by `mode=`:

**`mode='capacity'` (what `notebook_export_cell.py` uses)** — the final deployable model:
the budget-re-tuned static policy (`best_static_under_budget(...)`, val-tuned) plus
budget-pressure triage (`gate='pressure'`, `triage_k=K_FINAL`), enforced by a binding
review budget. Decisions honour the budget; reviews the policy wanted but couldn't grant
(overflow) go to the profit-optimal binary fallback and their damage is booked in the
`overload_cost` bucket. `capacity_constrained_decisions` in `fraudcore` replicates the
notebook's `CapacityConstrainedEnv`/`CapacityAblationEnv.step` exactly (verified to match
the env's rollout profit to the dollar).

**`mode='unconstrained'`** — the headline amount-conditioned PPO policy (`cab_a4`) with
unlimited review capacity; `overload_cost` is always 0. Use only if you actually provision
enough analysts to meet its ~22–27% review rate.

### Reviewers and budget are the same number

In capacity mode the model's review budget and the dashboard's staffing are tied:

```
budget_frac = reviewers × reviews_per_reviewer_per_day / txns_per_day
```

Pass **`budget_frac`** and the reviewer count is derived; or pass **`reviewers`** and the
budget is derived (reviewers wins if both are given). `txns_per_day` is read from the test
frame's `TransactionDT` span (≈2,878/day on IEEE-CIS), matching the notebook. So 6 analysts
× 100/day ÷ 2,878 ≈ 20.8% (binding), and the dashboard's `reviewers`, daily capacity, and
the model's budget all agree. If capacity ≥ volume the budget is non-binding and
`review_budget_frac` is capped at 1.0.

### Profit drops when the budget binds — that's correct

At a binding budget, profit is legitimately lower than unconstrained (e.g. ~$46k at a 20%
budget vs ~$67k unconstrained on the real data) because you review fewer transactions than
the policy wants. Deploy the capacity-bound system only if review capacity actually binds;
if you can staff to the unconstrained review rate, deploy `mode='unconstrained'` and keep
the extra profit. At 8 analysts (≈27.8% capacity) the budget barely binds — the policy's
own review demand sits below capacity, so overload ≈ 0 and the two modes nearly coincide.

### One dashboard caveat (unchanged behaviour)

The `capacity_utilization` KPI divides span-total reviews by *daily* capacity, so it reads
high (≈ number of days in the span). The per-day "Daily Review Load" chart is the correct
view and matches the staffing. The dashboard file is kept byte-identical to the original,
so this pre-existing quirk is documented rather than patched.

## Customising the policy comparison

Pass `comparison_policies=` to `write_dashboard_outputs` to control the rows, e.g. to use
the doctor's "Manual vs Grid Search vs RL" labels:

```python
comparison_policies=[
    ("Manual (accept all)", "global", (0.5, 0.5)),
    ("Grid search",         "global", (gs_T_low, gs_T_high)),
    ("RL (amount-conditioned)", "amountcond",
         (_policy, Z_LOGA_MEAN, Z_LOGA_STD, MIN_T_LOW, MAX_T_HIGH)),
]
```

---

## Live reviewer control (re-tune staffing without retraining PPO)

The amount-conditioned PPO policy is the expensive, **frozen** artifact. The only pieces that
depend on the reviewer count are (1) the budget-re-tuned static base and (2) the budget-pressure
triage coefficient — both are validation-side grid searches with no gradient steps. So the
dashboard can re-tune the *entire deployed system* for a new staffing level in roughly a second.

**How to enable it.** Re-run the export cell (`notebook_export_cell.py`). It now snapshots the
validation set and the frozen PPO policy into `dashboard_outputs/deployment_bundle.npz`. When that
bundle is present (and the run is capacity-mode), the sidebar shows a **Reviewers** number field.
Changing it triggers a live re-tune:

```
reviewers ─► budget_frac = reviewers × reviews_per_day / txns_per_day
          ─► best_static_under_budget(frozen_ppo, …)   # 54 val rollouts
          ─► tune_triage_k(best_a, …)                  # ~9 val rollouts
          ─► capacity_constrained_decisions(best_a, …) on test   # 1 rollout
          ─► every KPI / chart / comparison updates
```

PPO is never touched. Results are cached per reviewer count, so repeated values are instant.

## What the fixes in this version address

- **Capacity utilization** now compares *daily* review load to *daily* capacity (it previously
  divided span-total reviews by daily capacity, reading ~3000% when the system was simply binding).
- **Policy comparison** now uses the **frozen `cab_a4`** for the "RL (amount-conditioned)" row
  (the true Stage-2 policy), and capacity-mode runs add an explicit
  "Deployed (capacity, N reviewers)" row that matches the executive summary, so the constraint's
  cost is visible instead of implied.
- **Naming**: the score is described as the **hybrid (denoising AE + CatBoost)** classifier output
  everywhere in the UI. The `catboost_risk_score` column name is kept for pipeline compatibility.

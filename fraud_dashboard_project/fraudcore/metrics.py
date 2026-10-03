"""Financial metrics and classification counts, matching the notebook conventions.

`profit_and_gain` reproduces compute_financial_metrics' profit / no-fraud / oracle /
profit-gain. The confusion counts use the notebook's F1 convention where a reviewed
transaction is scored as correctly classified (reviewer accuracy prev=1), so
precision/recall reconcile with the notebook.
"""
import numpy as np

from .costs import transaction_cost, DEFAULT_COST_PARAMS


def profit_and_gain(d, y, A, cp=DEFAULT_COST_PARAMS):
    """Net profit and normalised Profit Gain (Yildirim).

    model_profit = -sum(transaction_cost)
    no_fraud     = accept-everything profit
    oracle       = block-all-fraud, accept-all-legit profit
    profit_gain  = (model - no_fraud) / (oracle - no_fraud)
    """
    pr, ltv, flm, rc, prev = cp['pr'], cp['ltv'], cp['flm'], cp['rc'], cp['prev']
    A = np.asarray(A, float); y = np.asarray(y, int)
    model_profit = -float(transaction_cost(d, y, A, cp).sum())
    no_fraud = -float((np.where(y == 1, flm * A, -pr * A)).sum())
    oracle = -float((np.where(y == 1, 0.0, -pr * A)).sum())
    denom = oracle - no_fraud
    pg = (model_profit - no_fraud) / denom if denom != 0 else float('nan')
    return dict(model_profit=model_profit, no_fraud_profit=no_fraud,
                oracle_profit=oracle, profit_gain=pg)


def confusion_counts(d, y):
    """Binary fraud confusion using the trinary decisions and the notebook's F1 rule.

    accept -> predicted legit; reject -> predicted fraud; review -> predicted = truth
    (reviewer accuracy prev=1). So:
        TP = reject&fraud + review&fraud      FP = reject&legit
        FN = accept&fraud                     TN = accept&legit + review&legit
    """
    d = np.asarray(d, int); y = np.asarray(y, int)
    y_pred = np.where(d == 0, 0, np.where(d == 1, 1, y))
    tp = int(((y_pred == 1) & (y == 1)).sum())
    fp = int(((y_pred == 1) & (y == 0)).sum())
    fn = int(((y_pred == 0) & (y == 1)).sum())
    tn = int(((y_pred == 0) & (y == 0)).sum())
    return dict(tp=tp, fp=fp, tn=tn, fn=fn)


def classification_rates(cm):
    """precision / recall / false_positive_rate from a confusion-count dict."""
    tp, fp, tn, fn = cm['tp'], cm['fp'], cm['tn'], cm['fn']
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    return dict(precision=precision, recall=recall, false_positive_rate=fpr)

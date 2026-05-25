import numpy as np
from sklearn.metrics import accuracy_score, roc_curve


def compute_pad_metrics(labels: np.ndarray, preds: np.ndarray) -> dict:
    labels = 2 * labels - 1  # Convert {0, 1} to {-1, 1}

    tp = np.sum((preds == 1) & (labels == 1))
    tn = np.sum((preds == -1) & (labels == -1))
    fp = np.sum((preds == 1) & (labels == -1))
    fn = np.sum((preds == -1) & (labels == 1))
    apcer = fn / (fn + tn) if (fn + tn) > 0 else 0
    bpcer = fp / (fp + tp) if (fp + tp) > 0 else 0
    ace = (apcer + bpcer) / 2
    accuracy = accuracy_score(labels, preds)

    return {
        "accuracy": accuracy,
        "ace": ace,
        "apcer": apcer,
        "bpcer": bpcer,
    }

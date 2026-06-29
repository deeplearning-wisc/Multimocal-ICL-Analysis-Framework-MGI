from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional, Any

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


def task_decoding_probe(
    H_all: np.ndarray,   # (num_layers, N, d)
    y: np.ndarray,       # (N,)
    test_size=0.2,
    random_state=0,
    class_weight="balanced",
    max_iter=2000,
):
    L,N,_ = H_all.shape
    tr_idx, te_idx = train_test_split(
        np.arange(N), test_size=test_size, random_state=random_state, stratify=y
    )

    results = {}
    for l in range(L):
        X_train, X_test = H_all[l][tr_idx], H_all[l][te_idx]
        y_train, y_test = y[tr_idx], y[te_idx]

        probe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=max_iter, class_weight=class_weight, solver="lbfgs"
            )),
        ])
        probe.fit(X_train, y_train)
        acc = accuracy_score(y_test, probe.predict(X_test))
        results[l] = {"acc": float(acc), "probe": probe, "idx": (tr_idx, te_idx)}
    return results


def per_layer_accuracy(results: Dict[int, Any]):
    return sorted([(l, v["acc"]) for l, v in results.items()], key=lambda x: x[0])


def plot_task_decoding_curve(
    res: Dict,
    fig_path: Optional[str] = None,
    which: str = "AUROC",
    title: Optional[str] = None,
):
    """
    画出每个探针模型的层内曲线（默认 AUROC）
    """
    layers = res["layers"]
    plt.figure(figsize=(8, 4.8), dpi=160)
    for name, md in res["metrics"].items():
        if which not in md:
            continue
        ys = md[which]
        plt.plot(layers, ys, marker="o", linewidth=2, label=name)

    plt.xlabel("Layer")
    plt.ylabel(which)
    if title:
        plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    if fig_path:
        plt.tight_layout()
        plt.savefig(fig_path)
        plt.close()
    else:
        plt.show()
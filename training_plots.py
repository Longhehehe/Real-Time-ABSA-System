import os
from typing import Dict, List, Optional, Sequence, Tuple


def _get_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib is not available, skipping plot.")
        return None
    return plt


def save_loss_plot(
    epoch_losses: List[float],
    output_dir: str,
    filename: str = "training_loss_by_epoch.png",
    title: str = "Training Loss by Epoch",
    xlabel: str = "Epoch",
    ylabel: str = "Loss",
) -> Optional[str]:
    if not epoch_losses:
        print("No epoch losses found, skipping loss plot.")
        return None

    plt = _get_matplotlib()
    if plt is None:
        return None

    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, filename)
    epochs = list(range(1, len(epoch_losses) + 1))

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, epoch_losses, marker="o", linewidth=2)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.xticks(epochs)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()

    print(f"Saved loss plot to: {plot_path}")
    return plot_path


def save_metric_bars(
    metrics: Dict[str, float],
    output_dir: str,
    filename: str = "metrics_comparison.png",
    title: str = "Metrics Comparison",
    ylabel: str = "Score",
    metric_order: Optional[Sequence[str]] = None,
) -> Optional[str]:
    if not metrics:
        print("No metrics found, skipping metric chart.")
        return None

    plt = _get_matplotlib()
    if plt is None:
        return None

    if metric_order:
        labels = [label for label in metric_order if label in metrics]
    else:
        labels = list(metrics.keys())

    if not labels:
        print("No matching metrics for chart, skipping.")
        return None

    values = [metrics[label] for label in labels]

    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, filename)

    width = max(8, int(len(labels) * 1.5))
    plt.figure(figsize=(width, 6))
    bars = plt.bar(range(len(labels)), values, color="#4C78A8")

    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(range(len(labels)), labels, rotation=25, ha="right")

    if all(0.0 <= v <= 1.0 for v in values):
        plt.ylim(0, 1.0)

    for idx, val in enumerate(values):
        plt.text(idx, val + 0.01, f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()

    print(f"Saved metrics chart to: {plot_path}")
    return plot_path


def save_models_metrics_comparison(
    models_metrics: Dict[str, Dict[str, float]],
    output_dir: str,
    filename: str = "all_models_metrics_comparison.png",
    title: str = "Model Metrics Comparison",
    metric_order: Optional[Sequence[Tuple[str, str]]] = None,
) -> Optional[str]:
    if not models_metrics:
        print("No model metrics found, skipping comparison chart.")
        return None

    plt = _get_matplotlib()
    if plt is None:
        return None

    import numpy as np

    model_names = list(models_metrics.keys())
    if not model_names:
        print("No model names found, skipping comparison chart.")
        return None

    if metric_order is None:
        metric_order = [
            ("Mention F1", "mention_f1_macro"),
            ("Sentiment F1", "sentiment_f1_macro"),
            ("Sentiment F1 (samples)", "sentiment_f1_samples"),
            ("Combined F1", "combined_f1"),
        ]

    metric_labels = [label for label, _ in metric_order]
    metric_keys = [key for _, key in metric_order]

    num_models = len(model_names)
    num_metrics = len(metric_keys)
    x = np.arange(num_models)
    bar_width = min(0.8 / max(num_metrics, 1), 0.2)

    plt.figure(figsize=(max(10, num_models * 1.8), 6))

    for idx, (label, key) in enumerate(zip(metric_labels, metric_keys)):
        values = [models_metrics[m].get(key, 0.0) for m in model_names]
        offset = (idx - (num_metrics - 1) / 2) * bar_width
        plt.bar(x + offset, values, width=bar_width, label=label)

    plt.title(title)
    plt.ylabel("Score")
    plt.xticks(x, model_names, rotation=20, ha="right")
    plt.legend()

    all_values = [models_metrics[m].get(k, 0.0) for m in model_names for k in metric_keys]
    if all_values and all(0.0 <= v <= 1.0 for v in all_values):
        plt.ylim(0, 1.0)

    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, filename)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()

    print(f"Saved model comparison chart to: {plot_path}")
    return plot_path


def save_fold_scores_plot(
    fold_metrics: List[Dict[str, float]],
    output_dir: str,
    filename: str = "fold_scores.png",
    title: str = "Per-Fold Validation Scores",
) -> Optional[str]:
    """Plot key metrics per fold for ML models (substitute for training loss chart).

    Shows Combined Score, Mention F1-macro, Sentiment F1-samples, and AUC-ROC
    for each K-Fold split so training quality can still be assessed visually.
    """
    if not fold_metrics:
        print("No fold metrics found, skipping fold scores plot.")
        return None

    plt = _get_matplotlib()
    if plt is None:
        return None

    import numpy as np

    metric_cfg = [
        ("Combined Score",    "combined_score",         "#2196F3"),
        ("Mention F1-macro",  "mention_f1_macro",       "#4CAF50"),
        ("Sentiment F1-samp", "sentiment_f1_samples",   "#FF9800"),
        ("Mention AUC-ROC",   "mention_auc_roc",        "#9C27B0"),
        ("Sentiment AUC-ROC", "sentiment_auc_roc",      "#F44336"),
    ]

    n_folds = len(fold_metrics)
    x = np.arange(1, n_folds + 1)

    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, filename)

    plt.figure(figsize=(max(8, n_folds * 1.5), 6))
    for label, key, color in metric_cfg:
        values = [m.get(key, 0.0) for m in fold_metrics]
        plt.plot(x, values, marker="o", linewidth=2, label=label, color=color)

    plt.title(title)
    plt.xlabel("Fold")
    plt.ylabel("Score")
    plt.xticks(x, [f"Fold {i}" for i in x])
    plt.ylim(0, 1.05)
    plt.legend(loc="lower right", fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()

    print(f"Saved fold scores plot to: {plot_path}")
    return plot_path

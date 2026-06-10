"""Validation visualization utilities.

Handles the creation of scatterplots and dissipation evolution graphs
for the incremental models.
"""

import os
import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils import validation_metrics
from utils.model_utils import cleanup_memory


def generate_validation_plots(
    df_results: pd.DataFrame, output_dir: str, serie: str, fs: float
):
    """Generate validation plots: scatterplots and dissipation series."""
    os.makedirs(output_dir, exist_ok=True)

    target_cols = ["velocity_x", "velocity_y", "velocity_z"]
    pred_cols = ["velocity_predicted_x", "velocity_predicted_y", "velocity_predicted_z"]

    has_true_data = all(col in df_results.columns for col in target_cols)

    if has_true_data:
        print(f"\n{'='*60}")
        print("📊 GENERATING VALIDATION PLOTS")
        print(f"{'='*60}")

        df_clean = df_results.dropna(subset=target_cols + pred_cols)

        if len(df_clean) > 50000:
            print(
                f"  [RAM Protection] Reducing visualization from {len(df_clean)} to 50000 points on scatterplot."
            )
            df_plot = df_clean.sample(n=50000, random_state=42)
        else:
            df_plot = df_clean

        Y_true = df_plot[target_cols].values
        Y_pred = df_plot[pred_cols].values

        print("Generating 1:1 velocity scatterplots...")
        scatter_data = validation_metrics.generate_1to1_scatterplot_data(Y_pred, Y_true)

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        comp_names = ["u_x", "u_y", "u_z"]

        for idx, comp_name in enumerate(comp_names):
            stats = scatter_data[comp_name]
            ax = axes[idx]

            ax.scatter(stats["true"], stats["pred"], alpha=0.5, s=10)

            min_val = min(stats["true"].min(), stats["pred"].min())
            max_val = max(stats["true"].max(), stats["pred"].max())
            ax.plot(
                [min_val, max_val],
                [min_val, max_val],
                "r--",
                linewidth=2,
                label="Perfect 1:1",
            )

            ax.plot(
                [min_val, max_val],
                [
                    min_val * stats["slope"] + stats["intercept"],
                    max_val * stats["slope"] + stats["intercept"],
                ],
                "g-",
                linewidth=2,
                label="Fitted",
            )

            ax.set_xlabel(f"True {stats['label']}", fontsize=10)
            ax.set_ylabel(f"Predicted {stats['label']}", fontsize=10)
            ax.set_title(
                f"{stats['label']}\nR²={stats['r_squared']:.4f}, RMSE={stats['rmse']:.6f}",
                fontsize=10,
            )
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        scatter_path = os.path.join(output_dir, f"scatterplot_1to1_{serie}.png")
        plt.savefig(scatter_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        plt.close("all")
        print(f"  ✓ Saved to {scatter_path}")

        print("\n  Scatterplot Statistics:")
        for comp_name in comp_names:
            stats = scatter_data[comp_name]
            print(
                f"    {stats['label']:20s}: RMSE={stats['rmse']:.6f}, R²={stats['r_squared']:.6f}"
            )

        del df_clean, df_plot, Y_true, Y_pred
        cleanup_memory()
    else:
        print(
            f"\n[Warning] True velocity data not available for scatterplot generation."
        )

    print(f"{'='*60}\n")


def generate_dissipation_series_plot(
    blocks_indices: list,
    df_results: pd.DataFrame,
    output_dir: str,
    serie: str,
    fs: float,
):
    """Generate dissipation evolution plot across blocks."""
    target_cols = ["velocity_x", "velocity_y", "velocity_z"]
    pred_cols = ["velocity_predicted_x", "velocity_predicted_y", "velocity_predicted_z"]

    has_true_data = all(col in df_results.columns for col in target_cols)

    if not has_true_data:
        print("[Info] Skipping dissipation series plot (no true velocity data).")
        return

    print("Generating dissipation series plot across blocks...")

    epsilons_pred = []
    epsilons_true = []
    block_nums = []

    for block_idx, indices in enumerate(blocks_indices):
        block_df = df_results.iloc[indices]
        if len(block_df) < 100:
            continue

        Y_true = block_df[target_cols].dropna().values
        Y_pred = block_df[pred_cols].dropna().values

        if len(Y_true) == 0 or len(Y_pred) == 0:
            continue

        try:
            u_true_fluc = Y_true - np.mean(Y_true, axis=0)
            u_pred_fluc = Y_pred - np.mean(Y_pred, axis=0)

            ke_true = 0.5 * np.mean(np.sum(u_true_fluc**2, axis=1))
            ke_pred = 0.5 * np.mean(np.sum(u_pred_fluc**2, axis=1))

            epsilons_true.append(ke_true)
            epsilons_pred.append(ke_pred)
            block_nums.append(block_idx + 1)
        except Exception as e:
            print(
                f"  Warning: Could not calculate dissipation for block {block_idx + 1}: {e}"
            )
            continue

    if len(block_nums) == 0:
        print("[Warning] No valid blocks for dissipation analysis.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        block_nums, epsilons_true, "o-", label="True (Sonic)", linewidth=2, markersize=8
    )
    ax.plot(
        block_nums,
        epsilons_pred,
        "s-",
        label="Predicted (Model)",
        linewidth=2,
        markersize=8,
    )

    ax.set_xlabel("Block Number", fontsize=12)
    ax.set_ylabel("Turbulent Kinetic Energy (Proxy)", fontsize=12)
    ax.set_title(
        f"Energy Evolution Across Sequential Blocks - Serie {serie}", fontsize=13
    )
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    if len(epsilons_pred) > 1:
        jumps = np.abs(np.diff(epsilons_pred))
        mean_jump = np.mean(jumps)
        max_jump = np.max(jumps)

        info_text = f"Block-to-Block Continuity:\nMean Jump: {mean_jump:.6e}\nMax Jump: {max_jump:.6e}"
        ax.text(
            0.02,
            0.98,
            info_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"dissipation_series_{serie}.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    plt.close("all")
    print(f"  ✓ Saved to {plot_path}")
    cleanup_memory()

import json
import os
from typing import Dict, Any

import matplotlib.pyplot as plt
import seaborn as sns


def plot_auc_results(
    results_data: Dict[str, Any],
    output_dir: str,
    baseline_results_path: str = None,
):
    """
    Generate and save a professional ICML-style plot for AUC results.

    This function creates a plot of faithfulness vs. graph density, comparing the
    tested method against a baseline (e.g., 'greedy') and an ideal faithfulness line.

    Args:
        results_data (Dict[str, Any]): Dictionary containing the results of the evaluated method.
        output_dir (str): Directory to save the plot.
        baseline_results_path (str, optional): Path to the JSON results file for the baseline method.
    """
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.5)
    plt.figure(figsize=(10, 7))

    # Plot the results of the method that was just run
    percentages = results_data["results"].get("percentages", [])
    # Ensure percentages are plotted correctly on a log scale by excluding 0
    plot_percentages = [p if p > 0 else 1e-4 for p in percentages]
    faithfulnesses = results_data["results"].get("faithfulnesses", [])
    method_label = f"{results_data['evaluation_info']['method']} (AUC: {results_data['results']['auc_score']:.3f})"
    plt.plot(
        plot_percentages,
        faithfulnesses,
        marker="o",
        linestyle="-",
        label=method_label,
        zorder=3,
    )

    # Plot the baseline method results if a path is provided and the file exists
    if baseline_results_path and os.path.exists(baseline_results_path):
        try:
            with open(baseline_results_path, "r") as f:
                baseline_data = json.load(f)
            baseline_percentages = baseline_data["results"].get("percentages", [])
            baseline_plot_percentages = [
                p if p > 0 else 1e-4 for p in baseline_percentages
            ]
            baseline_faithfulnesses = baseline_data["results"].get("faithfulnesses", [])
            baseline_label = f"{baseline_data['evaluation_info']['method']} (AUC: {baseline_data['results']['auc_score']:.3f})"
            plt.plot(
                baseline_plot_percentages,
                baseline_faithfulnesses,
                marker="s",
                linestyle="--",
                label=baseline_label,
                color="gray",
                zorder=2,
            )
        except (json.JSONDecodeError, KeyError) as e:
            print(
                f"Warning: Could not load or parse baseline results file '{baseline_results_path}': {e}"
            )

    # Add a line for ideal faithfulness at y=1.0
    plt.axhline(
        y=1.0, color="r", linestyle=":", label="Ideal Faithfulness (1.0)", zorder=1
    )

    # Set plot title and labels
    plt.title(
        f"Faithfulness vs. Graph Density\nModel: {results_data['evaluation_info']['model']}, Task: {results_data['evaluation_info']['task']}"
    )
    plt.xlabel("Percentage of Edges in Graph (Log Scale)")
    plt.ylabel("Faithfulness Score (Pruned / Full Model)")
    plt.xscale("log")

    # Configure x-axis ticks for log scale clarity
    xtick_locations = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
    xtick_labels = ["0.1%", "0.5%", "1%", "2%", "5%", "10%", "20%", "50%", "100%"]
    plt.xticks(xtick_locations, xtick_labels)
    plt.minorticks_off()

    plt.ylim(bottom=min(0, plt.ylim()[0]))

    plt.legend()
    plt.tight_layout()

    # Save the plot to the output directory
    timestamp = results_data["evaluation_info"]["timestamp"]
    model, task, method, metric, pnr = (
        results_data["evaluation_info"]["model"],
        results_data["evaluation_info"]["task"],
        "+".join(sorted(results_data["evaluation_info"]["method"])),
        results_data["evaluation_info"]["relevant_metric"],
        results_data["evaluation_info"]["pnr"],
    )
    plot_filename = f"{model}_{task}_{method}_{metric}{f'_pnr{pnr}' if pnr >= 0 else ''}_{timestamp}_plot.png"
    plot_path = os.path.join(output_dir, "plots", plot_filename)
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path, dpi=300)
    plt.close()

    print(f"Plot saved to {plot_path}")
    return plot_path

#!/usr/bin/env python3
import re
import matplotlib.pyplot as plt
import argparse
import numpy as np

def parse_log_file(log_file_path):
    """
    Parses the training log file to extract metrics.
    """
    train_steps_list, train_losses_list, train_mean_ranks_list = [], [], []
    val_steps_list, val_acc1_list, val_acc10_list, val_acc100_list, val_mean_ranks_list = [], [], [], [], []

    # Regex for numeric training lines: e.g., " 12680 | loss 4.839 | mean-rank 50.8"
    train_numeric_pattern = re.compile(r"^\s*(\d+)\s*\|\s*loss\s*([nanINF\d.-]+)\s*\|\s*mean-rank\s*([nanINF\d.-]+)")
    # Regex for N/A training lines: e.g., " 12710 | loss N/A | mean-rank N/A (...)"
    train_na_pattern = re.compile(r"^\s*(\d+)\s*\|\s*loss N/A\s*\|\s*mean-rank N/A.*")
    # Regex for validation summary: e.g., "└─ val@1 0.380 | val@10 0.470 | val@100 0.680 | mean_rank 118.2"
    val_summary_pattern = re.compile(r"└─ val@1\s*([nanINF\d.-]+)\s*\|\s*val@10\s*([nanINF\d.-]+)\s*\|\s*val@100\s*([nanINF\d.-]+)\s*\|\s*mean_rank\s*([nanINF\d.-]+)")

    current_step_for_val = None # Stores the step number to associate with a validation block

    with open(log_file_path, 'r', encoding='utf-8') as f:
        for line_num, line_content in enumerate(f, 1):
            line = line_content.strip()

            # Try matching numeric training log line
            match_train_numeric = train_numeric_pattern.match(line)
            if match_train_numeric:
                step = int(match_train_numeric.group(1))
                current_step_for_val = step # This step might precede a validation block
                try:
                    loss = float(match_train_numeric.group(2))
                    mean_rank = float(match_train_numeric.group(3))
                    train_steps_list.append(step)
                    train_losses_list.append(loss)
                    train_mean_ranks_list.append(mean_rank)
                except ValueError:
                    print(f"Warning (L{line_num}): Could not parse float from training metrics: '{match_train_numeric.group(2)}' or '{match_train_numeric.group(3)}' in line: {line}")
                continue # Line processed

            # Try matching N/A training log line
            match_train_na = train_na_pattern.match(line)
            if match_train_na:
                step = int(match_train_na.group(1))
                current_step_for_val = step # This step might precede a validation block
                train_steps_list.append(step)
                train_losses_list.append(float('nan')) # Use NaN for plotting gaps
                train_mean_ranks_list.append(float('nan'))
                continue # Line processed

            # Try matching validation summary log line
            match_val_summary = val_summary_pattern.match(line)
            if match_val_summary:
                if current_step_for_val is not None:
                    try:
                        acc1 = float(match_val_summary.group(1))
                        acc10 = float(match_val_summary.group(2))
                        acc100 = float(match_val_summary.group(3))
                        mean_rank_val = float(match_val_summary.group(4))

                        val_steps_list.append(current_step_for_val)
                        val_acc1_list.append(acc1)
                        val_acc10_list.append(acc10)
                        val_acc100_list.append(acc100)
                        val_mean_ranks_list.append(mean_rank_val)
                    except ValueError:
                        print(f"Warning (L{line_num}): Could not parse float from validation metrics: {line}")
                else:
                    # This might happen if the log starts with a validation summary or format is unexpected
                    print(f"Warning (L{line_num}): Found validation summary but no preceding step to link it to: {line}")
                continue # Line processed

    return {
        "train_steps": np.array(train_steps_list, dtype=np.int32),
        "train_losses": np.array(train_losses_list, dtype=np.float32),
        "train_mean_ranks": np.array(train_mean_ranks_list, dtype=np.float32),
        "val_steps": np.array(val_steps_list, dtype=np.int32),
        "val_acc1": np.array(val_acc1_list, dtype=np.float32),
        "val_acc10": np.array(val_acc10_list, dtype=np.float32),
        "val_acc100": np.array(val_acc100_list, dtype=np.float32),
        "val_mean_ranks": np.array(val_mean_ranks_list, dtype=np.float32),
    }

def plot_metrics(data, output_image_path="training_curves.png"):
    """
    Plots the parsed training and validation metrics.
    """
    if data["train_steps"].size == 0 and data["val_steps"].size == 0:
        print("No data found to plot.")
        return

    fig, axs = plt.subplots(4, 1, figsize=(12, 16), sharex=True)
    fig.suptitle('Training and Validation Curves', fontsize=16)

    # Helper to check if data for a specific key is present and not all NaNs
    def has_valid_data(key):
        return data[key].size > 0 and not np.all(np.isnan(data[key]))

    # Plot 1: Training Loss
    if has_valid_data("train_losses"): # Check specific metric
        axs[0].plot(data["train_steps"], data["train_losses"], label="Training Loss", color='tab:blue', alpha=0.8)
    axs[0].set_ylabel("Loss")
    axs[0].set_title("Training Loss")
    axs[0].legend(loc='best')
    axs[0].grid(True, linestyle='--', alpha=0.7)

    # Plot 2: Training Mean Rank
    if has_valid_data("train_mean_ranks"): # Check specific metric
        axs[1].plot(data["train_steps"], data["train_mean_ranks"], label="Training Mean Rank", color='tab:orange', alpha=0.8)
    axs[1].set_ylabel("Mean Rank")
    axs[1].set_title("Training Mean Rank")
    axs[1].legend(loc='best')
    axs[1].grid(True, linestyle='--', alpha=0.7)

    # Plot 3: Validation Accuracies
    plotted_val_acc = False
    if has_valid_data("val_steps"): # Check if there are any validation steps
        if has_valid_data("val_acc1"):
            axs[2].plot(data["val_steps"], data["val_acc1"], label="Validation val@1", color='tab:green', marker='o', linestyle='-', markersize=5)
            plotted_val_acc = True
        if has_valid_data("val_acc10"):
            axs[2].plot(data["val_steps"], data["val_acc10"], label="Validation val@10", color='tab:red', marker='s', linestyle='-', markersize=5)
            plotted_val_acc = True
        if has_valid_data("val_acc100"):
            axs[2].plot(data["val_steps"], data["val_acc100"], label="Validation val@100", color='tab:purple', marker='^', linestyle='-', markersize=5)
            plotted_val_acc = True
    axs[2].set_ylabel("Accuracy")
    axs[2].set_title("Validation Accuracies")
    if plotted_val_acc: # Only show legend if something was plotted
        axs[2].legend(loc='best')
    axs[2].grid(True, linestyle='--', alpha=0.7)

    # Plot 4: Validation Mean Rank
    if has_valid_data("val_steps") and has_valid_data("val_mean_ranks"): # Check specific metric
        axs[3].plot(data["val_steps"], data["val_mean_ranks"], label="Validation Mean Rank", color='tab:brown', marker='x', linestyle='-', markersize=5)
    axs[3].set_ylabel("Mean Rank")
    axs[3].set_title("Validation Mean Rank")
    axs[3].legend(loc='best') # Safe to call legend even if no data, it just won't show
    axs[3].grid(True, linestyle='--', alpha=0.7)

    axs[3].set_xlabel("Training Steps")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title/label overlap
    plt.savefig(output_image_path)
    print(f"Plots saved to {output_image_path}")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse a training log file and plot metrics.")
    parser.add_argument("log_file", type=str, help="Path to the training log file.")
    parser.add_argument("--output_image", type=str, default="training_curves.png",
                        help="Filename for the saved plot image (default: training_curves.png).")

    args = parser.parse_args()

    parsed_data = parse_log_file(args.log_file)
    plot_metrics(parsed_data, args.output_image)
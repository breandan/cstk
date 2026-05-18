import re
import matplotlib.pyplot as plt
import sys
import argparse

def parse_log(log_path):
    train_steps = []
    train_losses = []
    val_steps = []
    val_losses = []

    with open(log_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Parse training loss line: "step  11310 | loss 0.8434 | ..."
        train_match = re.search(r"step\s+(\d+)\s+\|\s+loss\s+([\d.]+)", line)
        if train_match:
            step = int(train_match.group(1))
            loss = float(train_match.group(2))
            train_steps.append(step)
            train_losses.append(loss)

        # Parse validation block
        if "--- validation @ step" in line:
            val_step_match = re.search(r"step (\d+)", line)
            if val_step_match and i + 1 < len(lines):
                val_step = int(val_step_match.group(1))
                next_line = lines[i + 1].strip()
                # "val summary ... | mean_loss 2.4912 | ..."
                val_loss_match = re.search(r"mean_loss\s+([\d.]+)", next_line)
                if val_loss_match:
                    val_loss = float(val_loss_match.group(1))
                    val_steps.append(val_step)
                    val_losses.append(val_loss)
                i += 1  # Skip the val summary line (and any export line that follows)

        i += 1

    return train_steps, train_losses, val_steps, val_losses

def main():
    parser = argparse.ArgumentParser(description='Plot training and validation loss curves from the log')
    parser.add_argument('log_file', help='Path to the training log file')
    parser.add_argument('--output', '-o', default='loss_curves.png', help='Output plot filename (default: loss_curves.png)')
    parser.add_argument('--smooth', type=int, default=0, help='Optional: moving average window size for smoothing training loss (e.g. 20)')
    args = parser.parse_args()

    train_steps, train_losses, val_steps, val_losses = parse_log(args.log_file)

    print(f"Parsed {len(train_steps)} training points and {len(val_steps)} validation points")

    # Optional smoothing for the noisy training curve
    if args.smooth > 1 and len(train_losses) > args.smooth:
        import numpy as np
        train_losses = np.convolve(train_losses, np.ones(args.smooth)/args.smooth, mode='valid')
        train_steps = train_steps[args.smooth//2 : args.smooth//2 + len(train_losses)]

    plt.figure(figsize=(14, 7))

    # Training loss (light blue, semi-transparent)
    plt.plot(train_steps, train_losses, label='Training Loss',
             color='#1f77b4', alpha=0.6, linewidth=1)

    # Validation loss (red markers + line)
    if val_steps:
        plt.plot(val_steps, val_losses, label='Validation Loss',
                 color='#d62728', marker='o', markersize=7, linewidth=2.5)

    plt.xlabel('Training Step')
    plt.ylabel('Loss')
    plt.title('Training vs Validation Loss Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(args.output, dpi=250, bbox_inches='tight')
    print(f"Plot saved to: {args.output}")

    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_loss.py your_log.txt [-o output.png] [--smooth 20]")
        sys.exit(1)
    main()

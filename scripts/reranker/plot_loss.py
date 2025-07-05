import re
import matplotlib.pyplot as plt
import sys

if len(sys.argv) < 2:
    print("Usage: python plot_training.py <log_file>")
    sys.exit(1)

log_file = sys.argv[1]

# Regular expressions to match log patterns
loss_pattern = r"step\s+(\d+)\s+\|\s+loss\s+(\d+\.\d+)"
mrr_pattern = r"MRR@100 =\s+(\d+\.\d+)"

# Lists to store data
losses = []
mrrs = []

# Read and parse the log file
with open(log_file, 'r') as f:
    for line in f:
        # Match average loss
        match_loss = re.match(loss_pattern, line)
        if match_loss:
            step = int(match_loss.group(1))
            avg_loss = float(match_loss.group(2))
            losses.append((step, avg_loss))

        # Match validation MRR
        match_mrr = re.match(mrr_pattern, line)
        if match_mrr:
            # step = int(match_mrr.group(1))
            mrr = float(match_mrr.group(1))
            mrrs.append((step, mrr))

# Extract steps and values for plotting
steps_loss = [step for step, _ in losses]
avg_losses = [avg_loss for _, avg_loss in losses]
steps_mrr = [step for step, _ in mrrs]
mrr_values = [mrr for _, mrr in mrrs]

# Create the plot
fig, ax1 = plt.subplots()

# Plot average loss on the first y-axis
ax1.plot(steps_loss, avg_losses, label='Avg Loss', color='blue')
ax1.set_xlabel('Step')
ax1.set_ylabel('Average Loss', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')

# Create a second y-axis for validation MRR
ax2 = ax1.twinx()
ax2.plot(steps_mrr, mrr_values, label='Validation MRR', color='red', marker='o')
ax2.set_ylabel('Validation MRR', color='red')
ax2.tick_params(axis='y', labelcolor='red')

# Add legends
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

# Set the title and display the plot
plt.title('Training Loss and Validation MRR over Steps')
plt.show()
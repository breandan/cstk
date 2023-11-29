import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load your data
df = pd.read_csv('repair_timings_1701235209240.csv', sep=r'\s*,\s*')

# Plotting setup
plt.figure(figsize=(10, 6))
colors = {1: 'red', 2: 'green', 3: 'blue'}

y_label = 'Throughput'
# Scatter plot
for patch_size in df['Patch size'].unique():
    subset = df[df['Patch size'] == patch_size]
    plt.scatter(subset['Snippet length'], subset[y_label],
                color=colors[patch_size], label=f'Patch size {patch_size}')

    # Calculate and plot line of best fit
    m, b = np.polyfit(subset['Snippet length'], subset[y_label], 1)
    plt.plot(subset['Snippet length'], m * subset['Snippet length'] + b, color=colors[patch_size])

# Labels, Title, and Y-axis limit
plt.xlabel('Snippet length')
plt.ylabel(y_label)
plt.title('Time to find human repair vs. Snippet length with Lines of Best Fit')
# plt.ylim(0, 1000)

# Legend
plt.legend()

# Show plot
plt.show()

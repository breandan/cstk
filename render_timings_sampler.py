import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load your data
df = pd.read_csv('repair_timings_1701677682309.csv', sep=r'\s*,\s*')

# Plotting setup
plt.figure(figsize=(10, 6))
colors = {1: 'red', 2: 'green', 3: 'blue'}

yaxis = 'Time to find human repair (ms)'
# Scatter plot
for patch_size in df['Patch size'].unique():
    subset = df[df['Patch size'] == patch_size]
    plt.scatter(subset['Snippet length'], subset[yaxis],
                color=colors[patch_size], label=f'Patch size {patch_size}')

blue_subset = df[df['Patch size'] == 3]
plt.scatter(blue_subset['Snippet length'], blue_subset[yaxis], color=colors[3])

# Labels, Title, and setting Y-axis to logarithmic
plt.xlabel('Snippet length (lexical tokens)')
plt.ylabel(yaxis)
plt.title(f'{yaxis} over lexical snippet length')
# plt.yscale('log')
# plt.ylim(0, 10000)
# plt.xlim(0, 200)

# Legend
plt.legend()

# Show plot
plt.show()

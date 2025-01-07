import matplotlib.pyplot as plt
import numpy as np

def plot_interwoven_histograms(file1, file2):
    def parse_file(file_path):
        with open(file_path, 'r') as file:
            lines = file.readlines()

        data = [list(map(int, line.split())) for line in lines]

        num_buckets = 20

        buckets = [[] for _ in range(num_buckets)]

        for line in data:
            line_length = len(line)
            for i, value in enumerate(line):
                normalized_index = int(i / line_length * (num_buckets - 1))
                buckets[normalized_index].append(value)

        averages = [np.mean(bucket) if bucket else 0 for bucket in buckets]
        errors = [np.std(bucket) / np.sqrt(len(bucket)) if bucket else 0 for bucket in buckets]

        return averages, errors

    averages1, errors1 = parse_file(file1)
    averages2, errors2 = parse_file(file2)

    num_buckets = 20
    bucket_centers = np.linspace(0, 1, num_buckets)

    bar_width = 1 / (2 * num_buckets)

    plt.rcParams.update({
      "pgf.texsystem": "pdflatex",
      'font.family': 'serif',
      'text.usetex': True,
      'pgf.rcfonts': False,
    })
    plt.figure(figsize=(10, 6))
    plt.bar(bucket_centers - bar_width / 2, averages1, yerr=errors1, width=bar_width, color='b', edgecolor='black', alpha=0.7, capsize=5, label='Constrained')
    plt.bar(bucket_centers + bar_width / 2, averages2, yerr=errors2, width=bar_width, color='r', edgecolor='black', alpha=0.7, capsize=5, label='Unconstrained')

    plt.yscale('log')
    # plt.title('True next-token rank over normalized snippet positions (Constrained vs. Unconstrained)')
    plt.xlabel('Normalized Position')
    plt.ylabel('Log Averaged Rank')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    # plt.show()

    plt.savefig('../galoisenne/latex/figures/rank_cst_unc.pgf')

# Example usage:
# (on MakeMore.measureRankOfTrueNextTokenWithLBHConstraints())
# ./gradlew collectSummaryStats 2>&1 | tee next_tok_ranks.txt
# cat next_tok_ranks.txt | grep '^CRANK_IDX' >> scripts/rank_idx_cst.txt
# cat next_tok_ranks.txt | grep '^URANK_IDX' >> scripts/rank_idx_unc.txt

plot_interwoven_histograms('crank_idx.txt', 'urank_idx.txt')
import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.gridspec import GridSpec

tx_iteration_results = json.load(open("evaluation_results.json", "r"))

plt.figure(figsize=(18, 15))
plt.style.use('ggplot')

gs = GridSpec(2, 2, height_ratios=[1, 1], width_ratios=[3, 1])

iterations = [1, 2, 3, 4, 5, 6, 7, 10, 20, 30]
tx_counts = [1000, 2000, 5000, 10000]
trigger_correlation = np.zeros((len(iterations), len(tx_counts)))
relative_duration = np.zeros((len(iterations), len(tx_counts)))
attributes_fidelity = np.zeros((len(iterations), len(tx_counts)))
for i, iteration in enumerate(iterations):
    for j, tx_count in enumerate(tx_counts):
        key = f'SynSpansIterations{iteration}TxCount{tx_count}'
        trigger_correlation[i, j] = tx_iteration_results["trigger_correlation"][key]["avg"]
        relative_duration[i, j] = tx_iteration_results["relative_duration"][key]["avg"]
        

# 1. Trigger Correlation Heatmap
ax1 = plt.subplot(gs[0, 0])
im1 = sns.heatmap(trigger_correlation, annot=True, fmt=".3f", cmap="Blues", 
                 vmin=0.7, vmax=0.85, ax=ax1,
                 xticklabels=tx_counts, yticklabels=iterations,
                 cbar_kws={'label': 'F1 Score'})
ax1.set_title('Trigger Correlation (higher is better)', fontweight='bold', fontsize=14)
ax1.set_xlabel('Trace Count', fontsize=12)
ax1.set_ylabel('Epoch', fontsize=12)

# 2. Relative Duration Heatmap
ax2 = plt.subplot(gs[1, 0])
im2 = sns.heatmap(relative_duration, annot=True, fmt=".3f", cmap="Reds_r", 
                 vmin=0, vmax=1, ax=ax2,
                 xticklabels=tx_counts, yticklabels=iterations,
                 cbar_kws={'label': 'Wasserstein Distance'})
ax2.set_title('Relative Duration (lower is better)', fontweight='bold', fontsize=14)
ax2.set_xlabel('Trace Count', fontsize=12)
ax2.set_ylabel('Epoch', fontsize=12)

head_samples = [5, 10, 15, 20, 30, 40, 50, 75, 100, 150]
head_trigger_correlation_avg = np.zeros(len(head_samples))
head_trigger_correlation_std = np.zeros(len(head_samples))
head_relative_duration_avg = np.zeros(len(head_samples))
head_relative_duration_std = np.zeros(len(head_samples))
for i, sample in enumerate(head_samples):
    key = f'HeadBasedTraces{sample}'
    head_trigger_correlation_avg[i] = tx_iteration_results["trigger_correlation"][key]["avg"]
    head_trigger_correlation_std[i] = tx_iteration_results["trigger_correlation"][key]["std"]
    head_relative_duration_avg[i] = tx_iteration_results["relative_duration"][key]["avg"]
    head_relative_duration_std[i] = tx_iteration_results["relative_duration"][key]["std"]

# 1. Head-based Traces - Trigger Correlation
ax4 = plt.subplot(gs[0, 1])
ax4.errorbar(head_samples, head_trigger_correlation_avg, 
            yerr=head_trigger_correlation_std, fmt='o-', color='#3274A1', 
            linewidth=2, markersize=6, capsize=4, ecolor='#1A3E5A')
ax4.set_ylim(0.9, 1.02)
ax4.set_xlabel('Sample Size')
ax4.set_ylabel('F1 Score')
ax4.set_title('Head-Based Traces\nTrigger Correlation', fontweight='bold')
ax4.grid(True, linestyle='--', alpha=0.7)

# 2. Head-based Traces - Relative Duration
ax5 = plt.subplot(gs[1, 1])
ax5.errorbar(head_samples, head_relative_duration_avg, 
            yerr=head_relative_duration_std, fmt='o-', color='#E1812C', 
            linewidth=2, markersize=6, capsize=4, ecolor='#8B4513')
ax5.set_ylim(0, 0.7)  # Increased to accommodate error bars
ax5.set_xlabel('Sample Size')
ax5.set_ylabel('Wasserstein Distance')
ax5.set_title('Head-Based Traces\nRelative Duration', fontweight='bold')
ax5.grid(True, linestyle='--', alpha=0.7)

plt.suptitle('GenT Experiments: Epoch/Trace Count and Head-Based Traces', 
             fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.grid(False)
plt.show()

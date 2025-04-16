import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def plot_ctgan_dim(gs, results):
    ctgan_dims = ["128", "128_128", "256", "256_256"]
    trigger_correlation = np.zeros(len(ctgan_dims))
    relative_duration_avg = np.zeros(len(ctgan_dims))
    relative_duration_std = np.zeros(len(ctgan_dims))
    for i, dim in enumerate(ctgan_dims):
        key = f'SynSpansCTGANDim{dim}'
        trigger_correlation[i] = results["trigger_correlation"][key]["avg"]
        relative_duration_avg[i] = results["relative_duration"][key]["avg"]
        relative_duration_std[i] = results["relative_duration"][key]["std"]

    ax1 = plt.subplot(gs[0, 0])
    bars1 = ax1.bar(ctgan_dims, trigger_correlation, color='#3274A1', alpha=0.8)
    ax1.set_ylim(0.9, 1.05)
    ax1.set_xlabel('Model Dimensions')
    ax1.set_ylabel('F1 Score')
    ax1.set_title('Trigger Correlation (higher is better)', fontweight='bold')
    ax1.set_xticks(ctgan_dims)
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{height:.4f}',
                ha='center', va='bottom', rotation=0, fontsize=9)

    ax2 = plt.subplot(gs[0, 1])
    bars2 = ax2.bar(ctgan_dims, relative_duration_avg, yerr=relative_duration_std, 
                capsize=5, color='#E1812C', alpha=0.8, 
                error_kw=dict(ecolor='black', lw=1, capsize=5, capthick=1))
    ax2.set_ylim(0, 0.4)
    ax2.set_xlabel('Model Dimensions')
    ax2.set_ylabel('Wasserstein Distance')
    ax2.set_title('Relative Duration (lower is better)', fontweight='bold')
    ax2.set_xticks(ctgan_dims)
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{height:.4f}',
                ha='center', va='bottom', rotation=0, fontsize=9)

if __name__ == "__main__":
    plt.figure(figsize=(18, 8))
    plt.style.use('ggplot')
    gs = GridSpec(1, 2)

    ctgan_dim_results = json.load(open("evaluation_results.json", "r"))
    plot_ctgan_dim(gs, ctgan_dim_results)


    plt.suptitle('GenT Experiment Results: Dimension', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

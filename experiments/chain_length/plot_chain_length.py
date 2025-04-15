import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def plot_chain_length(gs, results):
    chain_lengths = [2, 3, 4, 5]
    trigger_correlation = np.zeros(len(chain_lengths))
    relative_duration_avg = np.zeros(len(chain_lengths))
    relative_duration_std = np.zeros(len(chain_lengths))
    for i, length in enumerate(chain_lengths):
        key = f'SynSpansChainLength{length}'
        trigger_correlation[i] = results["trigger_correlation"][key]["avg"]
        relative_duration_avg[i] = results["relative_duration"][key]["avg"]
        relative_duration_std[i] = results["relative_duration"][key]["std"]

    ax1 = plt.subplot(gs[0, 0])
    bars1 = ax1.bar(chain_lengths, trigger_correlation, color='#3274A1', alpha=0.8)
    ax1.set_ylim(0.9, 1.05)  
    ax1.set_xlabel('Chain Length')
    ax1.set_ylabel('F1 Score')
    ax1.set_title('Trigger Correlation (higher is better)', fontweight='bold')
    ax1.set_xticks(chain_lengths)
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{height:.4f}',
                ha='center', va='bottom', rotation=0, fontsize=9)

    ax2 = plt.subplot(gs[0, 1])
    bars2 = ax2.bar(chain_lengths, relative_duration_avg, yerr=relative_duration_std, 
                capsize=5, color='#E1812C', alpha=0.8, 
                error_kw=dict(ecolor='black', lw=1, capsize=5, capthick=1))
    ax2.set_ylim(0, 0.3)
    ax2.set_xlabel('Chain Length')
    ax2.set_ylabel('Wasserstein Distance')
    ax2.set_title('Relative Duration (lower is better)', fontweight='bold')
    ax2.set_xticks(chain_lengths)
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{height:.4f}',
                ha='center', va='bottom', rotation=0, fontsize=9)


if __name__ == "__main__":
    plt.figure(figsize=(18, 8))
    plt.style.use('ggplot')
    gs = GridSpec(1, 2)

    chain_length_results = json.load(open("evaluation_results.json", "r"))
    plot_chain_length(gs, chain_length_results)

    plt.suptitle('GenT Experiment Results: Chain Length', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

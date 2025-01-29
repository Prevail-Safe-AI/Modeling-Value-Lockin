import json
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple
plt.style.use('ggplot')

clustered_path = './data/analysis/cpp-cluster/81-81-aligned.json'
# clustered_path = './data/analysis/cpp-cluster-run-ICL3/60-60-aligned.json'
with open(clustered_path, 'r') as f:
    data = json.load(f)

cluster_labels = [[item["label"] for item in snapshot["content"]] for snapshot in data]
num_clusters = max(max(labels) for labels in cluster_labels) + 1

composition_data = np.zeros((len(data), num_clusters))
for i, labels in enumerate(cluster_labels):
    for label in labels:
        composition_data[i, label] += 1

def make_composition_plot(data: np.ndarray, top_n_clusters: int = 9, smooth_window: int = 64) -> None:
    fig = plt.figure(figsize=(10, 7.5))
    
    # Dual y-axis plot
    ax2 = fig.add_subplot(111)
    ax1 = ax2.twinx()
    
    cluster_sizes = np.sum(data, axis=0)
    top_clusters = np.argsort(cluster_sizes)[-top_n_clusters:]
    
    def smooth(ts: np.ndarray) -> np.ndarray:
        head_mean = np.mean(ts[:smooth_window//16])
        tail_mean = np.mean(ts[-smooth_window//16:])
        exponential_kernel = np.exp(-np.linspace(-1, 1, smooth_window)**2)
        padded_ts = np.concatenate([np.ones(smooth_window*2) * head_mean, ts, np.ones(smooth_window*2) * tail_mean])
        return np.convolve(padded_ts, exponential_kernel / np.sum(exponential_kernel), mode='same')[smooth_window*2:-smooth_window*2]
    
    id2name = {
        1099: "Research Integrity",
        770: "Gene-Environment Interaction",
        233: "Neuroscience",
        368: "Self-Awareness",
        479: "Epigenetics",
        577: "Consciousness",
        521: "Honesty and Epistemics",
        432: "General CogSci",
        597: "Bayesianism",
    }
    
    constituents = []
    constituent_labels = []
    for cluster_id in top_clusters:
        constituents.append(smooth(data[:, cluster_id]))
        if cluster_id in id2name:
            constituent_labels.append(f"Topic {cluster_id}: {id2name[cluster_id]}")
        else:
            constituent_labels.append(f"Topic {cluster_id}")
        # ax1.plot(smooth(data[:, cluster_id]), label=f"Cluster {cluster_id}")
    
    constituents.append(smooth(np.sum(data, axis=1) - np.sum(data[:, top_clusters], axis=1)))
    constituent_labels.append("Less Popular Topics")
    # ax1.plot(smooth(np.sum(data, axis=1) - np.sum(data[:, top_clusters], axis=1)), label="Other")
    
    colors = ["#126168","#0f8b8d","#478f74","#7e935b","#ec9a29","#db7c26","#ca5d22","#b93f1e","#a8201a","#143642"][::-1]
    constituent_labels = constituent_labels[::-1]
    constituents = constituents[::-1]
    ax1.stackplot(range(len(data)), *constituents, labels=constituent_labels, colors=colors, alpha=0.3) 
    ax1.set_xlim((0, len(data)))
    ax1.set_ylim((0, 100))
    ax1.set_ylabel("Percentage of Knowledge Base Items")
    ax1.grid(False)
    
    shannon_entropy = []
    for i in range(len(data)):
        proportions = data[i] / np.sum(data[i])
        proportions = proportions[proportions > 0]
        shannon_entropy.append(-np.sum(proportions * np.log(proportions)))
    
    ax2.plot(smooth(shannon_entropy), label="Shannon Diversity Index of Topic Composition", color='black', linewidth=3, linestyle='--')
    ax2.set_xlim((0, len(data)))
    ax2.set_ylim((0, 3.5))
    ax2.set_ylabel("Diversity (Shannon Entropy)")
    
    ax2.legend(loc='lower left')
    ax1.legend(loc='upper right')
    plt.xlabel("Time")
    plt.title("Topic Composition Over Time")
    plt.savefig(clustered_path.replace('.json', '.pdf'))


if __name__ == '__main__':
    make_composition_plot(composition_data)
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import precision_recall_curve

def plot_pr_curve(pr_data_dict, title, save_path):
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8, 6))
    
    for label, data in pr_data_dict.items():
        scores = np.array(data["scores"])
        matches = np.array(data["matches"])
        if len(matches) == 0:
            continue
        # Sklearn precision recall curve
        # matches: 1 for TP, 0 for FP
        precision, recall, _ = precision_recall_curve(matches, scores)
        plt.plot(recall, precision, label=label, lw=2)
        
    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(loc="lower left")
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_f1_bar_chart(f1_data, title, save_path):
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 6))
    
    classes = list(f1_data.keys())
    f1_scores = list(f1_data.values())
    
    sns.barplot(x=classes, y=f1_scores, palette="viridis")
    plt.xlabel("Classes", fontsize=12)
    plt.ylabel("F1-Score", fontsize=12)
    plt.title(title, fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.ylim([0.0, 1.05])
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

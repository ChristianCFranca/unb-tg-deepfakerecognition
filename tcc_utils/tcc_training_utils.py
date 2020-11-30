from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

def plot_ROC(preds, labels, model_name):
    
    fpr_tpr_auc_score = []
    for pred in preds:
        score = roc_auc_score(preds[1].numpy(), preds[0][:, 1].numpy()).item()
        fpr, tpr, thresholds = roc_curve(preds[1], preds[0][:, 1])

        fpr_tpr_auc_score.append([fpr, tpr, score])

    fig, ax = plt.subplots(1,1, figsize=(8,8))

    ax.plot([0,1],[0,1], linestyle='--')

    for fts_data, label in zip(fpr_tpr_auc_score, labels):
        ax.plot(fts_data[0], fts_data[1], label=f'{model_name} {label} (AUC={fts_data[2]:.4f})')
        
    ax.set_xticks([-0.2,  0. ,  0.2,  0.4,  0.6,  0.8,  1. ,  1.2])
    ax.set_yticks([-0.2,  0. ,  0.2,  0.4,  0.6,  0.8,  1. ,  1.2])
    ax.set_xlim(-0.05, 1.05)
    ax.set_xlabel("False Positive Rate (FPR)", fontsize=13)
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("True Positive Rate (TPR)", fontsize=13)
    
    ax.legend(loc='lower right', fontsize=13)
    ax.set_title("ROC Curve", fontsize=15)


    plt.show()
    return score, thresholds
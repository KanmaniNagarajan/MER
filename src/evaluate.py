import torch
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from data_preprocessing import get_data_loaders
from model import EmotionRecognitionModel
from config import *
from utils import main_logger as logger
import json

def evaluate_model(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for videos, audios, labels in test_loader:
            videos, audios = videos.to(DEVICE), audios.to(DEVICE)
            outputs = model(audios, videos)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
    
    return np.array(all_preds), np.array(all_labels), np.array(all_probs)

def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(RESULTS_DIR, 'confusion_matrix.png'))
    plt.close()

def plot_roc_curve(y_true, y_score):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    n_classes = len(EMOTIONS)
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true == i, y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    plt.figure(figsize=(10, 8))
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label=f'{EMOTIONS[i]} (AUC = {roc_auc[i]:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(RESULTS_DIR, 'roc_curve.png'))
    plt.close()

def main():
    _, _, test_loader = get_data_loaders()
    model = EmotionRecognitionModel().to(DEVICE)
    model.load_state_dict(torch.load(os.path.join(RESULTS_DIR, 'best_model.pth')))
    
    y_pred, y_true, y_score = evaluate_model(model, test_loader)
    
    # Calculate and log metrics
    report = classification_report(y_true, y_pred, target_names=EMOTIONS, output_dict=True)
    logger.info(f"Classification Report:\n{json.dumps(report, indent=2)}")
    
    # Log recognition rate per emotion and overall
    logger.info("Recognition Rate per emotion:")
    for emotion in EMOTIONS:
        logger.info(f"{emotion}: {report[emotion]['recall']*100:.2f}%")
    logger.info(f"Overall Recognition Rate: {report['accuracy']*100:.2f}%")

    # Calculate and log F1-score
    f1_weighted = f1_score(y_true, y_pred, average='weighted')
    logger.info(f"Weighted F1-score: {f1_weighted:.4f}")
    logger.info("F1-score per emotion:")
    for emotion in EMOTIONS:
        logger.info(f"{emotion}: {report[emotion]['f1-score']:.4f}")
    
    # Plot and save confusion matrix
    plot_confusion_matrix(y_true, y_pred, EMOTIONS)
    logger.info("Confusion matrix saved to results directory")
    
    # Calculate and plot ROC curve
    plot_roc_curve(y_true, y_score)
    logger.info("ROC curve saved to results directory")
    
    # Calculate and log ROC AUC
    roc_auc = roc_auc_score(y_true, y_score, multi_class='ovr', average='weighted')
    logger.info(f"Weighted ROC AUC: {roc_auc:.4f}")

if __name__ == "__main__":
    main()

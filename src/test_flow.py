
import torch
import torch.nn as nn
import torch.optim as optim
import time
import signal
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.exceptions import UndefinedMetricWarning
import warnings
from data_preprocessing import get_data_loaders
from model import EmotionRecognitionModel
from config import *
from utils import main_logger as logger

# Suppress the UndefinedMetricWarning
warnings.filterwarnings('ignore', category=UndefinedMetricWarning)

def get_present_classes(y_true):
    """Get the unique classes present in the data."""
    unique_classes = np.unique(y_true)
    present_emotions = [EMOTIONS[i] for i in unique_classes]
    return unique_classes, present_emotions

def plot_confusion_matrix(y_true, y_pred, classes, save_path):
    """Plot confusion matrix with only present classes."""
    unique_classes, present_emotions = get_present_classes(y_true)
    
    cm = confusion_matrix(y_true, y_pred, labels=unique_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=present_emotions, 
                yticklabels=present_emotions)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_roc_curves(y_true, y_score, classes, save_path):
    """Plot ROC curves for present classes only."""
    unique_classes, present_emotions = get_present_classes(y_true)
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    plt.figure(figsize=(10, 8))
    for idx, class_idx in enumerate(unique_classes):
        try:
            # Create one-hot encoding for this class
            y_true_binary = (y_true == class_idx).astype(int)
            y_score_binary = y_score[:, class_idx]
            
            fpr[class_idx], tpr[class_idx], _ = roc_curve(y_true_binary, y_score_binary)
            roc_auc[class_idx] = auc(fpr[class_idx], tpr[class_idx])
            
            plt.plot(fpr[class_idx], tpr[class_idx], lw=2,
                    label=f'{present_emotions[idx]} (AUC = {roc_auc[class_idx]:.2f})')
        except Exception as e:
            logger.warning(f"Could not compute ROC curve for class {present_emotions[idx]}: {str(e)}")
            roc_auc[class_idx] = 0.0
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    return roc_auc

def test_flow(num_samples=2, timeout=300):
    logger.info(f"Starting test flow for {num_samples} samples with a {timeout} second timeout")
    
    # Create results directory if it doesn't exist
    results_dir = os.path.join(RESULTS_DIR, 'test_flow_results')
    os.makedirs(results_dir, exist_ok=True)

    metrics = {
        'training': {'loss': [], 'accuracy': [], 'predictions': [], 'true_labels': [], 'scores': []},
        'validation': {'loss': [], 'accuracy': [], 'predictions': [], 'true_labels': [], 'scores': []}
    }

    try:
        start_time = time.time()

        train_loader, val_loader, test_loader = get_data_loaders()
        if train_loader is None:
            logger.error("Failed to create data loaders. Aborting test flow.")
            return

        model = EmotionRecognitionModel().to(DEVICE)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        
        # Training phase
        logger.info("Starting training phase...")
        model.train()
        
        for batch_idx, (videos, audios, labels) in enumerate(train_loader):
            if batch_idx >= num_samples:
                break

            videos = videos.to(DEVICE)
            audios = audios.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(audios, videos)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Calculate accuracy and store predictions
            _, predicted = torch.max(outputs.data, 1)
            metrics['training']['loss'].append(loss.item())
            metrics['training']['predictions'].extend(predicted.cpu().numpy())
            metrics['training']['true_labels'].extend(labels.cpu().numpy())
            metrics['training']['scores'].extend(torch.softmax(outputs, dim=1).detach().cpu().numpy())

            logger.info(f"Training Sample {batch_idx+1} completed. Loss: {loss.item():.4f}")

        # Calculate training metrics
        train_accuracy = accuracy_score(metrics['training']['true_labels'], metrics['training']['predictions'])
        logger.info(f"Training Accuracy: {train_accuracy:.4f}")

        # Validation phase
        logger.info("Starting validation phase...")
        model.eval()
        
        with torch.no_grad():
            for batch_idx, (videos, audios, labels) in enumerate(val_loader):
                if batch_idx >= num_samples:
                    break

                videos = videos.to(DEVICE)
                audios = audios.to(DEVICE)
                labels = labels.to(DEVICE)

                outputs = model(audios, videos)
                loss = criterion(outputs, labels)

                # Calculate accuracy and store predictions
                _, predicted = torch.max(outputs.data, 1)
                metrics['validation']['loss'].append(loss.item())
                metrics['validation']['predictions'].extend(predicted.cpu().numpy())
                metrics['validation']['true_labels'].extend(labels.cpu().numpy())
                metrics['validation']['scores'].extend(torch.softmax(outputs, dim=1).detach().cpu().numpy())

                logger.info(f"Validation Sample {batch_idx+1} completed. Loss: {loss.item():.4f}")

        # Calculate validation metrics
        val_accuracy = accuracy_score(metrics['validation']['true_labels'], metrics['validation']['predictions'])
        logger.info(f"Validation Accuracy: {val_accuracy:.4f}")

        # Generate and save evaluation metrics
        for phase in ['training', 'validation']:
            if len(metrics[phase]['predictions']) > 0:
                # Get present classes
                unique_classes, present_emotions = get_present_classes(
                    metrics[phase]['true_labels'])
                
                logger.info(f"{phase.capitalize()} phase - Present emotions: {present_emotions}")
                
                # Classification Report
                report = classification_report(
                    metrics[phase]['true_labels'],
                    metrics[phase]['predictions'],
                    labels=unique_classes,
                    target_names=present_emotions,
                    output_dict=True,
                    zero_division=0
                )
                
                with open(os.path.join(results_dir, f'{phase}_classification_report.json'), 'w') as f:
                    json.dump(report, f, indent=4)
                
                # Confusion Matrix
                plot_confusion_matrix(
                    np.array(metrics[phase]['true_labels']),
                    np.array(metrics[phase]['predictions']),
                    EMOTIONS,
                    os.path.join(results_dir, f'{phase}_confusion_matrix.png')
                )
                
                # ROC Curves
                roc_auc = plot_roc_curves(
                    np.array(metrics[phase]['true_labels']),
                    np.array(metrics[phase]['scores']),
                    EMOTIONS,
                    os.path.join(results_dir, f'{phase}_roc_curves.png')
                )
                
                # Save metrics summary
                metrics_summary = {
                    'accuracy': accuracy_score(
                        metrics[phase]['true_labels'],
                        metrics[phase]['predictions']
                    ),
                    'average_loss': np.mean(metrics[phase]['loss']),
                    'present_emotions': present_emotions,
                    'roc_auc': {EMOTIONS[i]: auc for i, auc in roc_auc.items() if i in unique_classes}
                }
                
                with open(os.path.join(results_dir, f'{phase}_metrics_summary.json'), 'w') as f:
                    json.dump(metrics_summary, f, indent=4)

        end_time = time.time()
        logger.info(f"Test flow completed successfully in {end_time - start_time:.2f} seconds")
        logger.info(f"Results saved in: {results_dir}")

    except Exception as e:
        logger.error(f"An error occurred during test flow: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
    finally:
        if 'signal' in locals():
            signal.alarm(0)

if __name__ == "__main__":
    test_flow()
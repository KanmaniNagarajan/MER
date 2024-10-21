import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from data_preprocessing import get_data_loaders
from model import EmotionRecognitionModel
from config import *
from utils import main_logger as logger
from sklearn.metrics import classification_report, f1_score

def test_flow(num_batches=5):
    """
    Test the entire flow of the emotion recognition system for a specified number of batches.
    
    Args:
    num_batches (int): Number of batches to process for testing.
    """
    logger.info(f"Starting test flow for {num_batches} batches")

    # Get data loaders
    train_loader, val_loader, test_loader = get_data_loaders()
    logger.info("Data loaders created successfully")

    # Initialize model, loss function, and optimizer
    model = EmotionRecognitionModel().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    logger.info(f"Model initialized on device: {DEVICE}")

    # Test training phase
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    all_preds = []
    all_labels = []

    for batch_idx, (videos, audios, labels) in enumerate(tqdm(train_loader, desc="Training")):
        if batch_idx >= num_batches:
            break

        videos, audios, labels = videos.to(DEVICE), audios.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(audios, videos)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        train_total += labels.size(0)
        train_correct += predicted.eq(labels).sum().item()

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        logger.info(f"Batch {batch_idx+1}/{num_batches}, Loss: {loss.item():.4f}")

    train_accuracy = 100. * train_correct / train_total
    logger.info(f"Training Accuracy: {train_accuracy:.2f}%")

    # Calculate recognition rate per emotion and overall
    report = classification_report(all_labels, all_preds, target_names=EMOTIONS, output_dict=True)
    logger.info("Recognition Rate per emotion:")
    for emotion in EMOTIONS:
        logger.info(f"{emotion}: {report[emotion]['recall']*100:.2f}%")
    logger.info(f"Overall Recognition Rate: {report['accuracy']*100:.2f}%")

    # Calculate F1-score (weighted and per-class)
    f1_weighted = f1_score(all_labels, all_preds, average='weighted')
    logger.info(f"Weighted F1-score: {f1_weighted:.4f}")
    logger.info("F1-score per emotion:")
    for emotion in EMOTIONS:
        logger.info(f"{emotion}: {report[emotion]['f1-score']:.4f}")

    # Test validation phase
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    all_val_preds = []
    all_val_labels = []

    with torch.no_grad():
        for batch_idx, (videos, audios, labels) in enumerate(tqdm(val_loader, desc="Validation")):
            if batch_idx >= num_batches:
                break

            videos, audios, labels = videos.to(DEVICE), audios.to(DEVICE), labels.to(DEVICE)
            outputs = model(audios, videos)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()

            all_val_preds.extend(predicted.cpu().numpy())
            all_val_labels.extend(labels.cpu().numpy())

            logger.info(f"Validation Batch {batch_idx+1}/{num_batches}, Loss: {loss.item():.4f}")

    val_accuracy = 100. * val_correct / val_total
    logger.info(f"Validation Accuracy: {val_accuracy:.2f}%")

    # Calculate recognition rate per emotion and overall for validation
    val_report = classification_report(all_val_labels, all_val_preds, target_names=EMOTIONS, output_dict=True)
    logger.info("Validation Recognition Rate per emotion:")
    for emotion in EMOTIONS:
        logger.info(f"{emotion}: {val_report[emotion]['recall']*100:.2f}%")
    logger.info(f"Overall Validation Recognition Rate: {val_report['accuracy']*100:.2f}%")

    # Calculate F1-score for validation (weighted and per-class)
    val_f1_weighted = f1_score(all_val_labels, all_val_preds, average='weighted')
    logger.info(f"Validation Weighted F1-score: {val_f1_weighted:.4f}")
    logger.info("Validation F1-score per emotion:")
    for emotion in EMOTIONS:
        logger.info(f"{emotion}: {val_report[emotion]['f1-score']:.4f}")

    # Test inference on a few samples
    model.eval()
    with torch.no_grad():
        for batch_idx, (videos, audios, labels) in enumerate(tqdm(test_loader, desc="Testing")):
            if batch_idx >= num_batches:
                break

            videos, audios, labels = videos.to(DEVICE), audios.to(DEVICE), labels.to(DEVICE)
            outputs = model(audios, videos)
            _, predicted = torch.max(outputs, 1)

            for i in range(len(labels)):
                logger.info(f"Sample {i+1}: True label: {EMOTIONS[labels[i]]}, Predicted: {EMOTIONS[predicted[i]]}")

    logger.info("Test flow completed successfully")

if __name__ == "__main__":
    test_flow()

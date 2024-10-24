import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import logging
import json
import os
from data_preprocessing import get_data_loaders
from model import EmotionRecognitionModel
from config import *
from utils import main_logger as logger

def train_model():
    # Get data loaders
    train_loader, val_loader, _ = get_data_loaders()
    logger.info("Data loaders created successfully")

    # Initialize model, loss function, and optimizer
    model = EmotionRecognitionModel().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    logger.info(f"Model initialized on device: {DEVICE}")
    logger.info(f"Training started with {NUM_EPOCHS} epochs")

    best_val_accuracy = 0.0
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(NUM_EPOCHS):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, (videos, audios, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} - Training")):
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

            if batch_idx % 10 == 0:
                logger.info(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}")

        train_accuracy = 100. * train_correct / train_total
        train_losses.append(train_loss / len(train_loader))
        train_accuracies.append(train_accuracy)
        logger.info(f'Epoch {epoch+1}/{NUM_EPOCHS}, Train Loss: {train_loss/len(train_loader):.4f}, Train Accuracy: {train_accuracy:.2f}%')

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for videos, audios, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} - Validation"):
                videos, audios, labels = videos.to(DEVICE), audios.to(DEVICE), labels.to(DEVICE)
                outputs = model(audios, videos)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_accuracy = 100. * val_correct / val_total
        val_losses.append(val_loss / len(val_loader))
        val_accuracies.append(val_accuracy)
        logger.info(f'Validation Loss: {val_loss/len(val_loader):.4f}, Validation Accuracy: {val_accuracy:.2f}%')

        # Save the best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), os.path.join(RESULTS_DIR, 'best_model.pth'))
            logger.info(f"New best model saved with validation accuracy: {best_val_accuracy:.2f}%")

        # Save intermediate results
        intermediate_results = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accuracies': train_accuracies,
            'val_accuracies': val_accuracies,
        }
        with open(os.path.join(RESULTS_DIR, 'intermediate_results.json'), 'w') as f:
            json.dump(intermediate_results, f)

    logger.info("Training completed")
    return model

if __name__ == "__main__":
    trained_model = train_model()
    torch.save(trained_model.state_dict(), os.path.join(RESULTS_DIR, 'final_model.pth'))
    logger.info("Final model saved successfully")

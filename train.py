import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import recall_score, f1_score, roc_auc_score
import argparse
import os
from models import *
from datasets import *

def parse_args():
    parser = argparse.ArgumentParser(description="Train a model for power system stability classification")
    
    # Model and training parameters
    parser.add_argument("--model", type=str, default="cnn", choices=["cnn", "cnn_pairwise", "mlp", "transformer", "lstm"],
                        help="Model architecture to use")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size for training")
    parser.add_argument("--dataset", type=str, default="Node123_loop", help="Dataset name")
    parser.add_argument("--train_size", type=float, default=0.8, help="Proportion of data to use for training")
    parser.add_argument("--epochs", type=int, default=500, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.005, help="Learning rate")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save model checkpoints")
    parser.add_argument("--seed", type=int, default=20, help="Random seed for reproducibility")
    parser.add_argument("--device", type=str, default="", help="Device to use (leave empty for auto-detection)")
    
    return parser.parse_args()

def train(model, train_loader, test_loader, criterion, optimizer, epochs, device, model_path):
    best_train_loss = float('inf')
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
        train_loss = running_loss / len(train_loader)
        
        # Evaluation phase
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
        
        test_loss /= len(test_loader)
        
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
        
        # Save model if train loss improves
        if train_loss < best_train_loss:
            best_train_loss = train_loss
            torch.save(model.state_dict(), model_path)
            print(f"Model saved to {model_path}")

def evaluate(model, data_loader, device, name="Test"):
    model.eval()
    correct = 0
    total = 0
    
    labels_list = []
    predictions_list = []
    probabilities_list = []
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probability = torch.sigmoid(outputs)  # Convert logit to probability
            prediction = (probability > 0.5).float()
            total += labels.size(0)
            correct += (prediction == labels).sum().item()
            
            labels_list.extend(labels.cpu().numpy())
            predictions_list.extend(prediction.cpu().numpy())
            probabilities_list.extend(probability.cpu().numpy())
    
    accuracy = 100 * correct / total
    print(f"{name} Accuracy: {accuracy:.2f}%")
    
    # Calculate additional metrics
    labels_array = np.array(labels_list)
    predictions_array = np.array(predictions_list)
    probabilities_array = np.array(probabilities_list)
    
    recall = recall_score(labels_array, predictions_array)
    f1 = f1_score(labels_array, predictions_array)
    auc = roc_auc_score(labels_array, probabilities_array)
    
    print(f"{name} Recall: {recall:.4f}")
    print(f"{name} F1 Score: {f1:.4f}")
    print(f"{name} AUC: {auc:.4f}")
    
    return accuracy, recall, f1, auc

if __name__ == "__main__":
    args = parse_args()

    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Load dataset
    dataset = BinaryDataset(args.dataset)
    train_size = int(args.train_size * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size], 
        generator=torch.Generator().manual_seed(args.seed)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Initialize model based on argument
    if args.model == "cnn":
        model = CnnClassifier().to(device)
    elif args.model == "cnn_pairwise":
        model = CnnClassifierPairwise().to(device)
    elif args.model == "mlp":
        model = MLPClassifier(dataset.n_buses).to(device)
    elif args.model == "transformer":
        model = TransformerClassifier(dataset.n_buses).to(device)
    elif args.model == "lstm":
        model = LSTMClassifier(dataset.n_buses).to(device)
    else:
        raise ValueError(f"Unknown model type: {args.model}")
    
    print(f"Model architecture: {model}")
    
    # Define loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Train model
    print(f"Starting training for {args.epochs} epochs...")
    os.makedirs(f"{args.checkpoint_dir}/{args.dataset}", exist_ok=True)

    model_path = f"{args.checkpoint_dir}/{args.dataset}/{args.model}_model.pth"
    train(model, train_loader, test_loader, criterion, optimizer, args.epochs, device, model_path)
    
    # Load best model and evaluate
    model.load_state_dict(torch.load(model_path, weights_only=True))
    
    print("\nEvaluating on test set:")
    test_metrics = evaluate(model, test_loader, device, name="Test")
    
    print("\nEvaluating on training set:")
    train_metrics = evaluate(model, train_loader, device, name="Train")
    
    print("\nTraining complete!")
    
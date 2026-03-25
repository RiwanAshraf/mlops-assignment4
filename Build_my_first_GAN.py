#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import mlflow
import mlflow.pytorch

LEARNING_RATE = 0.005 
BATCH_SIZE    = 564  
EPOCHS        = 3
STUDENT_ID    = "202201726"   

# ── 1. Data ──────────────────────────────────
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.FashionMNIST(root='./data', train=True,  download=True, transform=transform)
test_dataset  = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader   = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)


# ── 2. Model ─────────────────────────────────
class FashionCNN(nn.Module):
    def __init__(self):
        super(FashionCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                            

            nn.Conv2d(32, 64, kernel_size=3, padding=1), 
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                          
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        return self.classifier(self.features(x))



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct    += (outputs.argmax(1) == labels).sum().item()
        total      += images.size(0)

    return total_loss / total, correct / total


def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss    = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            correct    += (outputs.argmax(1) == labels).sum().item()
            total      += images.size(0)

    return total_loss / total, correct / total


mlflow.set_tracking_uri("sqlite:///C:/Users/riwan/OneDrive/Desktop/MLops A2 - Copy/mlflow.db")
mlflow.set_experiment("Assignment3_Riwan")  

with mlflow.start_run():

    mlflow.set_tag("student_id", STUDENT_ID)
    mlflow.set_tag("model",      "FashionCNN")
    mlflow.set_tag("dataset",    "FashionMNIST")

    # --- Parameters ---
    mlflow.log_param("learning_rate", LEARNING_RATE)
    mlflow.log_param("batch_size",    BATCH_SIZE)
    mlflow.log_param("epochs",        EPOCHS)
    mlflow.log_param("optimizer",     "Adam")
    mlflow.log_param("dropout",       0.4)

    # --- Model / Loss / Optimizer ---
    model     = FashionCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- Training Loop ---
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss,   val_acc   = evaluate(model, test_loader, criterion)

 
        mlflow.log_metric("train_loss",     train_loss, step=epoch)
        mlflow.log_metric("train_accuracy", train_acc,  step=epoch)
        mlflow.log_metric("val_loss",       val_loss,   step=epoch)
        mlflow.log_metric("val_accuracy",   val_acc,    step=epoch)

        print(f"Epoch {epoch:02d}/{EPOCHS} | "
              f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.4f}")

    # --- Save model via MLflow PyTorch flavor ---
    mlflow.pytorch.log_model(model, artifact_path="model")
    print("Run complete — model saved to MLflow artifacts.")


    run_id = mlflow.active_run().info.run_id
    with open("model_info.txt", "w") as f:
        f.write(run_id)

    print(f"Run ID saved to model_info.txt: {run_id}")
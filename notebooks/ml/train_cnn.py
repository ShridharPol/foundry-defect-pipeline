"""
CNN Defect Classifier — MobileNetV2 transfer learning on casting inspection images.
Tracks experiment with MLflow.
"""

import os
import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# --- Config ---
DATA_DIR = "data/raw/casting/casting_data/casting_data"
MODEL_DIR = "mlflow/models"
BATCH_SIZE = 32
EPOCHS = 10
LR = 0.001
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(MODEL_DIR, exist_ok=True)

# --- Transforms ---
train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

if __name__ == '__main__':

    # --- Data ---
    train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=train_transforms)
    val_dataset   = datasets.ImageFolder(os.path.join(DATA_DIR, "test"),  transform=val_transforms)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"Classes:    {train_dataset.classes}")
    print(f"Train size: {len(train_dataset)} | Val size: {len(val_dataset)}")
    print(f"Device:     {DEVICE}")

    # --- Model ---
    model = models.mobilenet_v2(weights='IMAGENET1K_V1')
    model.classifier[1] = nn.Linear(model.last_channel, 2)
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    # --- Training ---
    mlflow.set_experiment("foundry_defect_detection")

    with mlflow.start_run(run_name="mobilenetv2_casting_classifier"):
        mlflow.log_params({
            "model":       "MobileNetV2",
            "epochs":      EPOCHS,
            "batch_size":  BATCH_SIZE,
            "learning_rate": LR,
            "img_size":    IMG_SIZE,
            "device":      str(DEVICE),
            "train_size":  len(train_dataset),
            "val_size":    len(val_dataset),
        })

        best_val_acc = 0.0

        for epoch in range(EPOCHS):
            # Train
            model.train()
            train_loss, train_correct = 0.0, 0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss    = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss    += loss.item() * inputs.size(0)
                train_correct += (outputs.argmax(1) == labels).sum().item()

            scheduler.step()
            train_acc  = train_correct / len(train_dataset)
            train_loss = train_loss    / len(train_dataset)

            # Validate
            model.eval()
            val_loss, val_correct = 0.0, 0
            all_preds, all_labels = [], []
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                    outputs  = model(inputs)
                    loss     = criterion(outputs, labels)
                    val_loss    += loss.item() * inputs.size(0)
                    val_correct += (outputs.argmax(1) == labels).sum().item()
                    all_preds.extend(outputs.argmax(1).cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

            val_acc  = val_correct / len(val_dataset)
            val_loss = val_loss    / len(val_dataset)

            mlflow.log_metrics({
                "train_loss": round(train_loss, 4),
                "train_acc":  round(train_acc,  4),
                "val_loss":   round(val_loss,   4),
                "val_acc":    round(val_acc,    4),
            }, step=epoch)

            print(f"Epoch {epoch+1:02d}/{EPOCHS} | "
                  f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.4f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), os.path.join(MODEL_DIR, "best_mobilenetv2.pth"))
                print(f"  ✓ New best — model saved")

        # --- Final report ---
        print("\nClassification Report:")
        print(classification_report(all_labels, all_preds, target_names=train_dataset.classes))

        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d',
                    xticklabels=train_dataset.classes,
                    yticklabels=train_dataset.classes, ax=ax)
        ax.set_title("Confusion Matrix — MobileNetV2")
        plt.tight_layout()
        cm_path = os.path.join(MODEL_DIR, "confusion_matrix_cnn.png")
        plt.savefig(cm_path)
        mlflow.log_artifact(cm_path)

        mlflow.log_metric("best_val_acc", round(best_val_acc, 4))
        mlflow.pytorch.log_model(model, "mobilenetv2_model")

        print(f"\nBest Val Accuracy: {best_val_acc:.4f}")
        print(f"Model saved to {MODEL_DIR}/best_mobilenetv2.pth")
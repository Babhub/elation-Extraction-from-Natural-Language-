"""
Training loop for multi-label relation extraction with early stopping.
"""

import torch
from torch import nn, optim
from eval import evaluate_metrics
import copy

def train_model(model, predict_fn, train_loader, val_loader, device,
                lr=1e-3, num_epochs=50, patience=5, save_path='best_model.pt'):
    
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()  # Multi-label classification
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_f1 = 0
    best_model_wts = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)

        # Validation
        model.eval()
        val_running_loss = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                val_loss = criterion(outputs, targets)
                val_running_loss += val_loss.item() * inputs.size(0)

        val_epoch_loss = val_running_loss / len(val_loader.dataset)
        val_losses.append(val_epoch_loss)

        # Evaluate metrics
        f1_weighted, _, _, _, _ = evaluate_metrics(model, val_loader, predict_fn, device)
        print(f"Epoch {epoch+1}: Train Loss={epoch_loss:.4f}, Val Loss={val_epoch_loss:.4f}, Val Weighted F1={f1_weighted:.4f}")

        # Early stopping
        if f1_weighted > best_val_f1:
            best_val_f1 = f1_weighted
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    # Load best model weights
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), save_path)
    print(f"Best model saved at {save_path} with weighted F1={best_val_f1:.4f}")

    return model, train_losses, val_losses

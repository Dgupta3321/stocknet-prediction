import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef


def evaluate(model, loader, device):
    """Run the model over a loader and return accuracy, F1 and MCC."""
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for x_val, y_val in loader:
            x_val = x_val.to(device)
            p = model(x_val).squeeze(-1).cpu().numpy()
            preds.extend(np.atleast_1d(p))
            targets.extend(np.atleast_1d(y_val.numpy()))

    preds_bin = (np.array(preds) > 0.5).astype(int)
    targets = np.array(targets).astype(int)
    return {
        "accuracy": accuracy_score(targets, preds_bin),
        "f1": f1_score(targets, preds_bin, zero_division=0),
        "mcc": matthews_corrcoef(targets, preds_bin),
    }


def train_model(model, train_loader, val_loader, epochs=20, lr=0.001, device="cpu"):
    """
    Train the model and return the validation metrics from the best epoch,
    selected on MCC rather than accuracy.

    Accuracy is a poor selection criterion here: next-day direction is close to
    balanced, so a model that collapses to always predicting "up" can score a
    respectable accuracy while carrying no information at all. MCC is zero for
    exactly that degenerate model.
    """
    model = model.to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    best = {"accuracy": 0.0, "f1": 0.0, "mcc": -1.0, "epoch": 0}

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            preds = model(x_batch).squeeze(-1)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        metrics = evaluate(model, val_loader, device)

        print(
            f"Epoch {epoch + 1:02d} | Train Loss: {train_loss / len(train_loader):.4f} "
            f"| Val Acc: {metrics['accuracy']:.4f} "
            f"| Val F1: {metrics['f1']:.4f} "
            f"| Val MCC: {metrics['mcc']:.4f}"
        )

        if metrics["mcc"] > best["mcc"]:
            best = {**metrics, "epoch": epoch + 1}

    print(
        f"Best epoch {best['epoch']:02d} | Acc {best['accuracy']:.4f} "
        f"| F1 {best['f1']:.4f} | MCC {best['mcc']:.4f}"
    )
    return best

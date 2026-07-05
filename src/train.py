import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from dataset import StockSequenceDataset
from models import StockMovementPredictor

def train_model(model, train_loader, val_loader, epochs=20, lr=0.001, device='cpu'):
    model = model.to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            preds = model(x_batch).squeeze()
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # Validation evaluation phase
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val = x_val.to(device)
                p = model(x_val).squeeze().cpu().numpy()
                val_preds.extend(p)
                val_targets.extend(y_val.numpy())
                
        val_preds_bin = (np.array(val_preds) > 0.5).astype(int)
        acc = accuracy_score(val_targets, val_preds_bin)
        mcc = matthews_corrcoef(val_targets, val_preds_bin)
        
        print(f"Epoch {epoch+1:02d} | Train Loss: {train_loss/len(train_loader):.4f} | Val Acc: {acc:.4f} | Val MCC: {mcc:.4f}")

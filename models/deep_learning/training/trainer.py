"""
Training and Validation Utilities

This module provides core functions for training an LSTM model for one epoch,
evaluating it on a validation/test set, and computing regression metrics
(MAE, RMSE, MAPE) on the original scale.
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error


def train_one_epoch(model, loader, optimizer, criterion, device, return_loss=False):
    """
    Perform one full pass over the training data.

    The model is set to training mode. For each batch, gradients are zeroed,
    a forward pass is computed, the loss is backpropagated, and the optimizer
    updates the weights.

    Args:
        model (nn.Module): The LSTM model to train.
        loader (DataLoader): DataLoader yielding (X_batch, y_batch).
        optimizer (torch.optim.Optimizer): Optimizer for weight updates.
        criterion: Loss function (e.g., MSELoss).
        device (torch.device): Device on which tensors are located.

    Returns:
        float: Average training loss over the entire epoch.
    """
    model.train()
    total_loss = 0.0

    for batch_X, batch_y in loader:
        # Data should be already on the GPU
        # batch_X, batch_y = batch_X.to(device), batch_y.to(device)

        optimizer.zero_grad()                 # Clear previous gradients
        preds = model(batch_X)                # Forward pass
        loss = criterion(preds, batch_y)      # Compute loss
        loss.backward()                       # Backpropagation
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()                      # Update weights
        
        if return_loss: total_loss += loss.item() * batch_X.size(0)

    if return_loss:
        avg_loss = total_loss / len(loader.dataset)
        return avg_loss

def validate(model, loader, criterion, device, return_preds = False, return_loss=True):
    """
    Evaluate the model on a validation or test set.

    The model is set to evaluation mode and gradients are disabled.
    Predictions and actual values are collected for metric computation.

    Args:
        model (nn.Module): The LSTM model to evaluate.
        loader (DataLoader): DataLoader yielding (X_batch, y_batch).
        criterion: Loss function (e.g., MSELoss).
        device (torch.device): Device on which tensors are located.

    Returns:
        tuple: (avg_loss, predictions, actuals)
            - avg_loss (float): Average loss over the dataset.
            - predictions (np.ndarray): Concatenated predictions.
            - actuals (np.ndarray): Concatenated ground truth values.
    """
    model.eval()
    total_loss = 0.0
    if return_preds:
        preds_list = []
        actuals_list = []

    with torch.no_grad():
        for batch_X, batch_y in loader:
            # batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            preds = model(batch_X)
            if return_loss:
                loss = criterion(preds, batch_y)
                total_loss += loss.item() * batch_X.size(0)

            if return_preds:
                preds_list.append(preds.cpu().numpy())
                actuals_list.append(batch_y.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    if return_preds: 
        preds_all = np.vstack(preds_list)
        actuals_all = np.vstack(actuals_list)
        if return_loss:
            return avg_loss, preds_all, actuals_all
        else: return preds_all, actuals_all
    else:
        return avg_loss


def compute_metrics(actuals, preds, target_scaler=None):
    """
    Compute standard regression metrics on the original (unscaled) scale.

    Args:
        actuals (np.ndarray): Ground truth values (already inverse-transformed).
        preds (np.ndarray): Predicted values (already inverse-transformed).
        target_scaler: Unused; kept for backward compatibility.

    Returns:
        tuple: (mae, rmse, mape)
            - mae (float): Mean Absolute Error.
            - rmse (float): Root Mean Squared Error.
            - mape (float): Mean Absolute Percentage Error.
    """
    mae = mean_absolute_error(actuals, preds)
    rmse = np.sqrt(mean_squared_error(actuals, preds))
    mape = mean_absolute_percentage_error(actuals, preds)
    return mae, rmse, mape
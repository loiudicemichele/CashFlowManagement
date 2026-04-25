"""
Final Model Training with Best Hyperparameters

This module trains the final LSTM model using the optimal hyperparameters
found during grid search. A chronological train/validation split is used
to monitor learning progress and perform early stopping via model checkpointing.
Training and validation loss histories are saved for later visualization.
"""

import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

from models.lstm_model import CashFlowLSTM
from training.trainer import train_one_epoch, validate


def final_training(X_train_t, y_train_t, best_params, config, device, output_dir):
    """
    Train the final LSTM model with the best hyperparameters.

    The full training data is split chronologically into training and validation
    sets (using `config.validation_split`). The model is trained for
    `config.final_epochs` epochs. After each epoch, validation loss is computed
    and the model state is saved if validation loss improves. The loss history
    is saved to `training_loss_history.csv` in the output directory.

    Args:
        X_train_t (torch.Tensor): Full training sequences tensor.
        y_train_t (torch.Tensor): Full training targets tensor.
        best_params (dict): Best hyperparameters from grid search.
        config: ExperimentConfig object containing final_epochs and validation_split.
        device (torch.device): Device to run training on.
        output_dir (str): Directory where the model and loss history will be saved.

    Returns:
        CashFlowLSTM: The trained model (with the best weights loaded).
    """
    print(f"[+] Starting Final Training with Best Params: {best_params}")

    # Chronological train/validation split
    split_idx = int(len(X_train_t) * (1 - config.validation_split))
    X_tr, y_tr = X_train_t[:split_idx], y_train_t[:split_idx]
    X_val, y_val = X_train_t[split_idx:], y_train_t[split_idx:]

    print(f"[+] Training samples: {len(X_tr)}, Validation samples: {len(X_val)}")

    # DataLoaders
    train_loader = DataLoader(
        TensorDataset(X_tr, y_tr),
        batch_size = best_params['batch_size'],
        shuffle = False
    )
    val_loader = DataLoader(
        TensorDataset(X_val, y_val),
        batch_size = best_params['batch_size'],
        shuffle = False
    )

    # Initialize model with best hyperparameters
    model = CashFlowLSTM(
        input_dim=X_train_t.shape[2],
        hidden_dim=best_params['hidden_dim'],
        num_layers=best_params['num_layers'],
        output_dim=config.n_outputs,
        dropout=best_params['dropout']
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=best_params['learning_rate'])
    criterion = torch.nn.MSELoss()

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_model_path = f"{output_dir}/best_lstm_model.pth"

    print(f"[+] Training for {config.final_epochs} epochs...")

    for epoch in range(config.final_epochs):
        # Training phase
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        # Validation phase
        val_loss = validate(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Model checkpointing
        is_best = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            is_best = "  <-- [BEST MODEL SAVED]"

        # Print progress every 5 epochs or when a new best model is found
        if (epoch + 1) % 5 == 0 or is_best:
            print(f"Epoch {epoch + 1:03d}/{config.final_epochs} | "
                  f"Train Loss (MSE): {train_loss:.6f} | "
                  f"Val Loss (MSE): {val_loss:.6f}{is_best}")

    print(f"\n[+] Final Training Completed. Best Validation Loss: {best_val_loss:.6f}")

    # Save loss history to CSV for plotting
    loss_df = pd.DataFrame({
        'epoch': range(1, config.final_epochs + 1),
        'train_loss': train_losses,
        'val_loss': val_losses
    })
    loss_csv_path = f"{output_dir}/training_loss_history.csv"
    loss_df.to_csv(loss_csv_path, index=False)
    print(f"[+] Loss history saved to: {loss_csv_path}")

    # Load the best weights back into the model before returning
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    print(f"[+] Best model weights loaded from: {best_model_path}")

    return model
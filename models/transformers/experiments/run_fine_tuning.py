# %% [markdown] # Chronos-2 Native Fine-Tuning
"""
This notebook fine-tunes Chronos-2 on domain-specific panel cash flow data
using the official pipeline.fit() API.

Instead of manually managing a training loop, we delegate the entire
tokenization, sliding-window cropping, and gradient update cycle to the
Chronos-2 native trainer. Two fine-tuning modes are available:
  - 'full': all model weights are updated (higher capacity, slower).
  - 'lora': only small adapter matrices are trained (faster, less memory).

Workflow:
    1. Load panel data in long format (all stores, all covariates).
    2. Convert to Chronos-2 List[dict] training format.
    3. Load the frozen pre-trained Chronos-2 pipeline.
    4. Fine-tune via pipeline.fit().
    5. Save the resulting pipeline for downstream inference.
"""

# %% [markdown] ## Imports and Environment Setup
"""
Import standard libraries, project modules, and the Chronos-2 pipeline class.
The project root is added to sys.path so all packages resolve correctly.
"""

import os
import sys
import numpy as np
import torch

sys.path.append(os.path.abspath('..'))

from chronos import Chronos2Pipeline
from config.experiment_config import ChronosConfig
from data.data_loader import load_covariates_df
from data.finetune_builder import build_chronos_training_dataset
from utils.helpers import set_seed_and_device

print("All modules imported successfully.\n")

# %% [markdown] ## Experiment Configuration
"""
Instantiate ChronosConfig and override the settings specific to fine-tuning.
Key parameters:
  - prediction_length: the forecast horizon the model is optimised for.
  - context_len: maximum history fed as input during training windows.
  - finetune_mode: 'lora' (recommended) or 'full'.
  - num_steps / learning_rate / batch_size: standard training hyperparameters.
    For LoRA the official notebook recommends lr=1e-4; for full fine-tuning lr=1e-5.
"""

config = ChronosConfig()

config.id_col    = 'store_id'
config.data_dir  = '../../../Datasets/data_partitioned'
config.model     = 'chronos-2-small'
config.model_name = os.path.join(config.model_dir, config.model)

# Fine-tuning hyperparameters
FINETUNE_MODE    = 'lora'     # 'lora' or 'full'
NUM_STEPS        = 1000
LEARNING_RATE    = 1e-4       # 1e-4 for lora, 1e-5 for full
BATCH_SIZE       = 32
LOGGING_STEPS    = 100

# Prediction horizon the model is specialised for during fine-tuning
# Should match the horizon used at inference time
config.prediction_length = 30

# Output directory for the fine-tuned model
finetuned_output_dir = os.path.join(config.model_dir, "chronos-2-cashflow-finetuned")
os.makedirs(finetuned_output_dir, exist_ok=True)

print(f"[+] Base model      : {config.model_name}")
print(f"[+] Fine-tune mode  : {FINETUNE_MODE}")
print(f"[+] Prediction len  : {config.prediction_length} steps")
print(f"[+] Output directory: {finetuned_output_dir}\n")

# %% [markdown] ## Set Seed and Device
"""
Fix all random seeds and select the hardware device for reproducibility.
"""

device = set_seed_and_device(config.seed, config.device)

# %% [markdown] ## Load Panel Data
"""
Load the raw long-format panel data.
load_covariates_df returns two DataFrames sorted by (store_id, date) and
validated for NaN values. We use only the training split for fine-tuning.
"""

train_df, _ = load_covariates_df(config)

print(f"[+] Training panel shape : {train_df.shape}")
print(f"[+] Unique stores        : {train_df[config.id_col].nunique()}\n")

# %% [markdown] ## Build Chronos-2 Training Dataset
"""
Convert the panel DataFrame into the List[dict] format expected by .fit().

Covariate split:
  - past_only: stochastic features whose future values are unknown at inference
    time (financial indicators, sales figures, etc.).
  - future_known: deterministic features whose future values are always known
    (calendar encodings, holiday flags). These go into both past_covariates
    (with real historical values) and future_covariates (with None, as a
    declaration of availability at inference time).
"""

# Stochastic covariates: all covariates minus the deterministic ones
past_only_cols = [
    col for col in config.covariate_cols
    if col not in config.deterministic_covariates
]

train_inputs = build_chronos_training_dataset(
    df=train_df,
    item_id_col=config.id_col,
    target_col=config.target_col,
    timestamp_col='date',
    past_only_covariate_cols=past_only_cols,
    future_known_covariate_cols=config.deterministic_covariates
)

# %% [markdown] ## Load Pre-Trained Chronos-2 Pipeline
"""
Load the base Chronos-2 pipeline from the local model directory.
The pipeline is placed on the selected device. No weights are modified yet.
"""

print("[+] Loading base Chronos-2 pipeline...")
pipeline = Chronos2Pipeline.from_pretrained(
    config.model_name,
    device_map=device
)
print("[+] Pipeline loaded successfully.\n")

# %% [markdown] ## Execute Fine-Tuning
"""
Call pipeline.fit() to perform the fine-tuning loop.

The method internally handles:
  - Sliding-window cropping of the full historical series into training samples.
  - Quantization of float values into discrete tokens via Chronos tokenizer.
  - Forward pass, cross-entropy loss, and gradient updates.
  - LoRA adapter injection when finetune_mode='lora'.

Returns a new pipeline instance with the fine-tuned weights; the base
pipeline object is not modified in-place.
"""

print(f"[+] Starting fine-tuning ({FINETUNE_MODE} mode, {NUM_STEPS} steps)...")

finetuned_pipeline = pipeline.fit(
    inputs=train_inputs,
    prediction_length=config.prediction_length,
    finetune_mode=FINETUNE_MODE,
    num_steps=NUM_STEPS,
    learning_rate=LEARNING_RATE,
    batch_size=BATCH_SIZE,
    logging_steps=LOGGING_STEPS,
)

print("\n[+] Fine-tuning completed successfully.")

# %% [markdown] ## Save Fine-Tuned Pipeline
"""
Persist the fine-tuned model weights and tokenizer configuration to disk.
Both components must be saved together so the pipeline can be reloaded
correctly via Chronos2Pipeline.from_pretrained(finetuned_output_dir).
"""

print(f"[+] Saving fine-tuned pipeline to: {finetuned_output_dir}")

finetuned_pipeline.model.save_pretrained(finetuned_output_dir)
finetuned_pipeline.save_pretrained(finetuned_output_dir + '/pipeline')
# finetuned_pipeline.tokenizer.save_pretrained(finetuned_output_dir)

print(f"[+] Model weights saved     : {finetuned_output_dir}")
print(f"[+] Tokenizer config saved  : {finetuned_output_dir}")
print("\n[+] Fine-tuning experiment completed successfully.")
# %% Test

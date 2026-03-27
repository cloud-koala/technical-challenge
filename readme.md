# How to run

Carpet Region (Task 1) detection and Feature Engineering plots (Task 2.1) are self-contained in the indicated jupyter notebooks in their respective sections.

For Structural Looseness Classification (Task 2.2), use our suggested method by running the following commands:

Training from scratch (optional):
`python -m src.training.train --config configs/order_spectrum_linear.yaml`

Inference only:
`python -m src.training.evaluate_external --config configs/order_spectrum_linear.yaml --artifacts artifacts/order_spectrum_linear --out external_test_predictions.csv`

Use environment at `env.yml`

# 1. Carpet Region Detection

See `carpet_regions.ipynb`

# 2. Structural Looseness Prediction

## 2.1 Feature Engineering plots

See `order_spectrum_features.ipynb`

## 2.2 Domain Knowledge + Data Driven Approach (Successful)

Algorithm:
```
# Simple baseline: linear classifier on order-spectrum features
#
# Produces probability scores via softmax(logits).


data:
  part3_root: data/part_3
  train_metadata: part_3_metadata.csv
  train_data_dir: data

orientation:
  output_order: [vertical, axial, horizontal]

preprocessing:
  feature_mode: order_spectrum

  downsample_hz: 400.0
  # Keep this relatively short so we always produce windows.
  window_seconds: 0.2
  step_seconds: 0.1

  z_norm: true

  rpm_min: 300.0
  rpm_max: 6000.0
  rpm_discrepancy_tol: 0.2
  rpm_top_k: 8
  rpm_harmonics: 5

  order_max: 10.0
  order_bins: 128
  order_log_power: true
  order_per_window_standardize: true

model:
  name: linear
  # NOTE: input_length defaults to preprocessing.window_size() for time mode
  # and preprocessing.order_bins for order_spectrum mode.
  in_channels: 3
  dropout: 0.0

  # Optional: condition on RPM
  rpm_conditioning: true
  rpm_embed_dim: 16

training:
  balance_train: false
  balance_val: true
  balance_test: true
  seed: 42
  batch_size: 256
  epochs: 20
  lr: 0.001
  optimizer: adamw
  weight_decay: 0.0
  label_smoothing: 0.0
  grad_clip_norm: null
  mixup_alpha: 0.0
  time_mask_prob: 0.0
  time_mask_ratio: 0.0
  early_stopping_patience: 0
  early_stopping_min_epochs: 0
  early_stopping_min_delta: 0.0
  num_workers: 0
  device: auto
  debug_plots: true
  debug_n_samples: 100
  debug_max_windows: 200  

protocol:
  name: loso
  split:
    train_ratio: 0.7
    val_ratio: 0.15
    test_ratio: 0.15

artifacts:
  output_dir: artifacts
  run_name: order_spectrum_linear
```

Leave-One-Subject-Out (LOSO) Cross-Validation Summary:
```
[
  {
    "heldout_sensor": "EZY2642",
    "val_f1_macro_best": 0.8675862068965516,
    "test_accuracy": 0.8125,
    "test_f1_macro": 0.8072672971627677
  },
  {
    "heldout_sensor": "PTA5611",
    "val_f1_macro_best": 0.8554865832430125,
    "test_accuracy": 0.95,
    "test_f1_macro": 0.949874686716792
  },
  {
    "heldout_sensor": "UKK6686",
    "val_f1_macro_best": 0.9624471913628541,
    "test_accuracy": 0.9353448275862069,
    "test_f1_macro": 0.9350734155487976
  },
  {
    "heldout_sensor": "VLQ4172",
    "val_f1_macro_best": 0.930849478390462,
    "test_accuracy": 0.925,
    "test_f1_macro": 0.9247058823529413
  }
]
```

Classification results:

|sample_id|rpm|pred_label|prob_structural_looseness|
|-|-|-|-|
|33542920-30ea-5844-861d-2c82d79087b8|1170.0|structural_looseness|0.7545|
|e057600e-3b4e-58ba-b8b8-357169ae6bf6|1800.0|structural_looseness|0.9262|
|01e98ad9-23c9-5986-ace0-4519bad71198|1785.0|healthy|0.3183|
|680bbcbf-b1c8-544d-8f80-bf763cdcd128|3573.0|structural_looseness|0.9967|
|2211750b-6672-5a94-bd40-cda811f69d01|2025.0|structural_looseness|0.9897|
|1dab1534-b8a8-5962-b01c-bff0782d54a9|3545.0|healthy|0.4841|
|9f3b933a-1bc3-5093-9dee-800cc03c6b1d|1590.0|structural_looseness|0.9812|



## 2.2 100% Data-Driven Approach (Failed)

We perform neural architecture search following a designed search space of simple and small convolution neural networks (CNNs) with a final MLP classification layer, executing over 350 trials of different architectures (varying kernel size, number of blocks, stem channel dimension) and hyperparameters. We do not employ manual feature engineering of any kind, only z-normalization of the original features. After training with the LOSO Cross-Validation we obtain the following results:
```
  {
    "heldout_sensor": "EZY2642",
    "val_f1_macro_best": 0.9833321758455449,
    "test_accuracy": 0.9166666666666666,
    "test_f1_macro": 0.916083916083916
  },
  {
    "heldout_sensor": "PTA5611",
    "val_f1_macro_best": 0.8674223401965707,
    "test_accuracy": 1.0,
    "test_f1_macro": 1.0
  },
  {
    "heldout_sensor": "UKK6686",
    "val_f1_macro_best": 0.9330357142857143,
    "test_accuracy": 0.9827586206896551,
    "test_f1_macro": 0.9827534939042522
  },
  {
    "heldout_sensor": "VLQ4172",
    "val_f1_macro_best": 0.5731590491964725,
    "test_accuracy": 0.9833333333333333,
    "test_f1_macro": 0.983328702417338
  }
```

The smallest network achieving these results is shown in the config file below.

```
data:
  part3_root: data/part_3
  train_metadata: part_3_metadata.csv
  train_data_dir: data

orientation:
  output_order: [vertical, axial, horizontal]

preprocessing:
  downsample_hz: 742.03797474182
  window_seconds: 0.0760876298581771
  step_seconds: null
  z_norm: true

model:
  name: conv_net
  in_channels: 3
  stem_channels: 32
  block_channels: [16, 16, 16, 16, 16]
  kernel_sizes: [3, 7, 33]
  dropout: 0.12555724603931073

training:
  seed: 42
  balance_train: true
  balance_val: true
  balance_test: true
  balance_seed: 42
  batch_size: 16
  epochs: 5
  lr: 0.0006966869590896839
  optimizer: adamw
  weight_decay: 2.3934518226431163e-05
  label_smoothing: 0.04352843601104618
  grad_clip_norm: 1.0
  mixup_alpha: 0.08572170917530572
  time_mask_prob: 0.11108994764880299
  time_mask_ratio: 0.19064439764498506
  num_workers: 0
  device: auto
  early_stopping_patience: 0.0
  early_stopping_min_epochs: 0.0
  early_stopping_min_delta: 0.0

protocol:
  name: loso
  split:
    train_ratio: 0.7
    val_ratio: 0.15
    test_ratio: 0.15
  loso:
    val_ratio_within_train: 0.15

artifacts:
  output_dir: artifacts
  run_name: small_conv_net_best
```
Despite promising results in the LOSO cross-validation of the train/val set, this approach did not generalize to out-of-distribution: classification results conflicted with our qualitative analysis.

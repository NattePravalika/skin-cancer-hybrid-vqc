# Hybrid CNN + Quantum VQC for Skin Cancer Detection

A hybrid classical-quantum deep learning pipeline that classifies dermoscopic skin lesion images as **benign** or **malignant**, combining a frozen CNN feature extractor with both a classical neural network and a variational quantum circuit (VQC), fused via ensemble prediction.

## Overview

This project explores whether quantum machine learning can meaningfully complement classical deep learning on a real-world medical imaging task. Rather than replacing the CNN backbone, the quantum circuit works alongside a classical MLP on the same extracted features, and their predictions are combined into a final ensemble verdict.

## Architecture

```
Input Image (224x224)
        │
        ▼
EfficientNet-B3 (frozen, pretrained)
        │
        ├──────────────┬────────────────┐
        ▼              ▼                
   Full 1536-dim   PCA (6 components)   
   features         + MinMax scaling    
        │              │                
        ▼              ▼                
   Classical MLP   Variational Quantum  
   (256→64→2)      Circuit (2 layers,   
                    6 qubits, PennyLane)
        │              │
        └──────┬───────┘
               ▼
        Ensemble (averaged softmax)
               │
               ▼
      Benign / Malignant + Confidence
```

- **Backbone:** EfficientNet-B3, pretrained on ImageNet, frozen (used purely as a feature extractor)
- **Classical branch:** Fully connected MLP on the full 1536-dim CNN feature vector
- **Quantum branch:** PCA-reduced features (6 dims) → angle-encoded into a 6-qubit variational circuit built with PennyLane, trained end-to-end via `torch` autodiff
- **Fusion:** Simple probability averaging across both branches at inference time

## Dataset

Trained and evaluated on dermoscopic skin lesion images, organized as:
```
skin_cancer_data/
    train/  → benign/, malignant/
    test/   → benign/, malignant/
```
Class imbalance is handled via a `WeightedRandomSampler` during training.

## Results

| Model                  | Accuracy | Precision | Recall | F1    | AUC   |
|------------------------|----------|-----------|--------|-------|-------|
| CNN + MLP (Classical)  | 88.7%    | —         | —      | —     | —     |
| CNN + VQC (Quantum)    | 76.5%    | —         | —      | —     | —     |
| **Hybrid Ensemble**    | **86.5%**| —         | —      | —     | —     |

*(Fill in precision/recall/F1/AUC from your saved checkpoint if you want the full table — they're stored in `skin_cancer_hybrid.pth`.)*

## Repository Contents

| File | Description |
|---|---|
| `Skin_Cancer_Detection_System.ipynb` | Full pipeline: training, evaluation, and inference (Colab-ready) |
| `skin_cancer_hybrid.pth` | Trained model checkpoint (MLP + VQC weights, PCA, scaler, class names, metrics) |
| `skin_cancer_predict_standalone.py` | Standalone script to run inference on a new image using the saved checkpoint |

## Usage

The notebook is structured in three independent stages:

1. **Train** — extracts CNN features, trains the classical MLP and quantum VQC, and saves the checkpoint to Google Drive, Colab's local storage, and as a downloadable `.pth` file.
2. **Evaluate** — reloads the saved checkpoint to regenerate confusion matrices and training curves, without retraining.
3. **Predict** — reloads the checkpoint and backbone, then runs inference on any uploaded image with a full breakdown across the classical, quantum, and ensemble predictions.

This means training only needs to happen once — evaluation and inference can be run independently at any time using the saved checkpoint.

## Tech Stack

- PyTorch, Torchvision (EfficientNet-B3)
- PennyLane (variational quantum circuit, `default.qubit` device)
- scikit-learn (PCA, MinMaxScaler, metrics)
- Google Colab (GPU runtime, T4)

## Disclaimer

This is a research/educational project and is **not a diagnostic tool**. Predictions should not be used for real medical decision-making.

## Author

Natte Pravalika

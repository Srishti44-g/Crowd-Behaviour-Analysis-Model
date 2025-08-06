# Crowd-Behaviour-Analysis-Model
# Hybrid Spatio-Temporal Graph Transformer for Crowd Behavior Analysis

**Date: August 6, 2025**

This repository contains the official implementation of the **Hybrid Spatio-Temporal Graph Transformer Network (HST-GTN)**, a state-of-the-art deep learning model designed to automatically understand and classify collective crowd behavior from video data.

The model can analyze video clips and predict the dominant crowd behavior, such as **"Normal"**, **"Panic"**, or **"Violent"**, making it a powerful tool for smart city surveillance, public safety, and event management.

---

## 🏛️ Architecture Overview

The HST-GTN is a multi-branch network that leverages the strengths of different specialized models to capture a comprehensive understanding of crowd dynamics. It processes visual, motion, and interaction data in parallel before fusing them intelligently for a final prediction.

```
Input Video Sequence
       │
       ├───────────────────┬───────────────────┐
       │                   │                   │
┌──────▼──────┐     ┌──────▼──────┐     ┌──────▼──────┐
│ Branch 1:   │     │ Branch 2:   │     │ Branch 3:   │
│ Visual      │     │ Interaction │     │ Global      │
│ Backbone    │     │ Module      │     │ Motion      │
│ (EfficientNet)│     │ (ST-GNN)    │     │ Module      │
│             │     │             │     │ (R(2+1)D)   │
└──────┬──────┘     └──────┬──────┘     └──────┬──────┘
       │                   │                   │
       └───────────┬───────┴───────────┘
                   │
          ┌────────▼────────┐
          │   Attention-    │
          │   based Fusion  │
          └────────┬────────┘
                   │
          ┌────────▼────────┐
          │ Prediction Head │
          │ (Classifier)    │
          └────────┬────────┘
                   │
                   ▼
      Output (e.g., "Normal", "Panic", "Violent")
```

---

## ✨ Features

* **Hybrid Multi-Branch Design:** Combines three expert modules for comprehensive analysis.
* **Interaction Modeling:** Uses a Spatio-Temporal Graph Neural Network (ST-GNN) to explicitly model social dynamics and interactions between individuals.
* **Motion Analysis:** Employs a 3D CNN to capture global motion patterns like running or fighting.
* **Visual Perception:** Leverages a powerful CNN backbone for rich visual feature extraction.
* **Attention-based Fusion:** Intelligently fuses the outputs of the three branches, focusing on the most relevant features for a given behavior.
* **End-to-End Training:** Built with PyTorch Lightning for clean, scalable, and reproducible training loops.
* **Experiment Tracking:** Integrated with Weights & Biases (W&B) for seamless logging and analysis of training runs.

---

## 📂 Project Structure

```
crowd_analysis/
├── data/                  # Raw video data (organized by class)
├── preprocessed_data/     # Stores object detection & tracking results
├── checkpoints/            # Saved model weights
├── src/
│   ├── data_loader.py     # PyTorch Datasets and Lightning DataModules
│   ├── models/
│   │   ├── cnn_branch.py
│   │   ├── gnn_branch.py
│   │   ├── motion_branch.py
│   │   └── hst_gtn.py     # Main hybrid model (LightningModule)
│   ├── train.py           # Main training script
│   └── evaluate.py        # Evaluation script
└── README.md              # This file
```

---

## 🚀 Getting Started

### 1. Prerequisites

Ensure you have Python 3.9+ and a CUDA-enabled GPU for optimal performance.

### 2. Installation

Clone the repository and install the required dependencies.

```bash
# Clone the repository
git clone <your-repo-url>
cd crowd_analysis

# Create and activate a virtual environment (recommended)
conda create -n crowd_analysis python=3.9
conda activate crowd_analysis

# Install dependencies
pip install torch torchvision pytorch-lightning torch-geometric ultralytics wandb
```

### 3. Data Pre-processing

The model requires pre-processed object tracking data.

1.  Organize your raw videos into subdirectories by class (e.g., `data/normal/`, `data/panic/`).
2.  Run the object detection and tracking script (you will need to create this script using a library like `ultralytics`). This script should iterate through your raw videos and save a corresponding `.json` file for each video in the `preprocessed_data/` directory. The JSON file should contain frame-by-frame bounding box and ID information for each person.

### 4. Training the Model

Once your data is pre-processed, you can start training.

1.  **Log in to W&B:**
    ```bash
    wandb login
    ```
2.  **Configure:** Adjust parameters like `num_classes`, `batch_size`, and data paths inside `src/train.py`.
3.  **Run Training:**
    ```bash
    python src/train.py
    ```
    Training progress, metrics, and model checkpoints will be automatically logged to your Weights & Biases project dashboard. The best model checkpoint will be saved locally in the `checkpoints/` directory.

### 5. Evaluating the Model

To evaluate your best-performing model on the test set:

1.  Update the `checkpoint_path` in `src/evaluate.py` to point to your saved `.ckpt` file.
2.  Run the evaluation script:
    ```bash
    python src/evaluate.py
    ```
    This will print the final performance metrics, such as validation loss and accuracy.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to open an issue or submit a pull request.

1.  Fork the Project
2.  Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3.  Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4.  Push to the Branch (`git push origin feature/AmazingFeature`)
5.  Open a Pull Request

---

## 📄 License

This project is distributed under the MIT License. See `LICENSE` for more information.

# 🩺 TB Chest X-Ray Image Classification with PyTorch

## Overview

This Jupyter notebook implements a deep learning model to classify chest X-ray images as Tuberculosis (TB) positive or negative using PyTorch. The project leverages Convolutional Neural Networks (CNNs) to automate TB detection from chest X-rays, aiding in early diagnosis and reducing the diagnostic workload for radiologists.

### Key Features

- **Data Source**: Utilizes the "TB_Chest_Radiography_Database" dataset available on Kaggle.
- **Model**: Employs DenseNet121 for binary classification (TB Positive vs TB Negative).
- **Techniques**:
  - Data loading and preprocessing using `torchvision`.
  - Train-validation-test split (80%-10%-10%).
  - Handling class imbalance with weighted loss functions.
  - Gradient accumulation for efficient training with large effective batch sizes.
  - Evaluation using accuracy, F1 score, and AUC-ROC metrics.

## Requirements

To run this notebook, you need the following dependencies:

- Python 3.10 or higher
- PyTorch
- Torchvision
- NumPy
- Matplotlib
- Scikit-learn
- tqdm

You can install the required packages using:

```bash
pip install torch torchvision numpy matplotlib scikit-learn tqdm
```

### Hardware
GPU acceleration is recommended (enabled via CUDA in the notebook).

### Dataset structure:
/path/to/dataset/
    ├── Normal/  # Folder containing TB Negative images
    └── Tuberculosis/  # Folder containing TB Positive images

---

## 🚀 Usage

1. **Clone or Download** the notebook and dataset.
2. Open the notebook in a **Jupyter environment** or **Google Colab**.
3. Ensure **GPU acceleration** is enabled (in Colab or locally).
4. Run the cells sequentially to:
   - Load and preprocess the data
   - Visualize sample images and class distribution
   - Train the **DenseNet121** model with gradient accumulation
   - Evaluate the model on validation and test sets
   - Plot **ROC curves** and other metrics

---

## 🧩 Key Code Sections

### 📂 Data Loading
- Uses `torchvision.datasets.ImageFolder` to load chest X-ray images.
- Applies transformations: resizing, normalization, and augmentations.

### 🧠 Model Training
- DenseNet121 backbone
- Gradient accumulation to handle large batch sizes on limited memory
- Optimizer: `RMSprop`
- Loss: Weighted Cross-Entropy (to handle class imbalance)

### 📊 Evaluation
- Metrics:
  - **Accuracy**
  - **F1 Score**
  - **AUC-ROC**
- Visualizations:
  - Confusion Matrix
  - ROC Curves
  - Loss/Accuracy plots

---

## 📈 Results

The model is evaluated using the following metrics:

- **Accuracy**: Proportion of correct predictions.
- **F1 Score**: Harmonic mean of precision and recall; robust to class imbalance.
- **AUC-ROC**: Area under the ROC curve indicating classifier performance.

Sample output includes:

- Training and validation loss/accuracy per epoch
- F1 and AUC scores on test set
- Plotted ROC curve

---

## 🧪 Potential Improvements

To further enhance performance:

- 🔁 **Reduce Batch Size**: Introduces gradient noise that can improve generalization.
- 📉 **Learning Rate Scheduling**: Train for more epochs with dynamic LR.
- 🧠 **Experiment with Larger Models**: Try ResNet, EfficientNet, etc.
- 🔍 **Hyperparameter Tuning**: Grid search or Bayesian optimization for better results.

---

## ✅ Conclusion

This notebook demonstrates a deep learning pipeline for TB detection using chest X-rays. It highlights the potential of AI in automated diagnostics, particularly for **resource-constrained healthcare settings**. Future work can focus on scaling the dataset and refining model robustness.

---

## 🙏 Acknowledgments

- **Dataset**: [TB Chest Radiography Database](https://data.mendeley.com/datasets/rscbjbr9sj/3)
- **Inspiration**:
  - World Health Organization (WHO) TB statistics
  - [radlines.org](https://www.radlines.org) for X-ray imagery

---

## 📜 License

This project is licensed under the **MIT License**. Feel free to use, modify, and distribute.


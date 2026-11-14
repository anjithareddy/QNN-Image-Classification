# Parameter-Efficient Image Classification via Variational Quantum Circuits (Hybrid QNN)

This repository contains the codebase and research for a Hybrid Quantum-Classical Neural Network designed for complex image classification tasks. Built for the Noisy Intermediate-Scale Quantum (NISQ) era, this project demonstrates how quantum entanglement can capture highly complex, non-linear decision boundaries with a drastically reduced parameter footprint compared to classical deep learning models.

## 🚀 Key Achievements

* **Massive Spatial Efficiency:** Achieved highly competitive binary classification accuracy on the Fashion-MNIST dataset operating with only **34 trainable parameters**, compared to a classical Convolutional Neural Network (CNN) baseline requiring **1,610 parameters**.
* **High-Dimensional Generalization:** Successfully extended the core 4-qubit architecture to process high-density RGB color images from the CIFAR-100 dataset.

## 🧠 Architecture Overview

The pipeline seamlessly bridges classical preprocessing and quantum machine learning utilizing PennyLane and PyTorch:

1. **Dimensionality Reduction (PCA):** Compresses raw classical image data (784 features for grayscale Fashion-MNIST, 3,072 features for RGB CIFAR-100) into a highly dense 4-dimensional latent space to bypass hardware qubit limitations.
2. **Data Encoding:** Utilizes **Angle Encoding** to map the continuous classical features into a 4-qubit quantum state via parameterized $R_y$ rotation gates.
3. **Variational Quantum Circuit (VQC):** Applies **Strongly Entangling Layers** (parameterized rotations followed by CNOT cascades) acting as the hidden weights of the network to capture complex feature correlations.
4. **Measurement & Output:** Measures the Pauli-Z expectation values and feeds them into a classical PyTorch linear layer to output final class probabilities.

## 📂 Repository Contents

* **"Quantum Neural Networks (QNNs) for Image Classification"** - The core Jupyter Notebook containing the complete end-to-end pipeline, including data loading, classical preprocessing, quantum circuit definition, classical CNN benchmarking, training loops, and evaluation metrics.

## 🛠️ Tech Stack

* **Quantum Framework:** PennyLane (Xanadu)
* **Deep Learning:** PyTorch
* **Machine Learning / Preprocessing:** Scikit-learn (PCA)
* **Data Visualization:** Matplotlib, Seaborn

## 📊 Datasets Evaluated

* **Fashion-MNIST:** Binary Classification (T-shirt/top vs. Trouser)
* **CIFAR-100:** Binary Classification (Apple vs. Aquarium Fish)

## 💻 Installation & Setup

**1. Clone the repository**

```bash
git clone https://github.com/your-username/hybrid-qnn-classification.git
cd hybrid-qnn-classification

```

**2. Create and activate a virtual environment**

* **Windows:**
```cmd
python -m venv qnn_env
qnn_env\Scripts\activate

```


* **macOS/Linux:**
```bash
python3 -m venv qnn_env
source qnn_env/bin/activate

```



**3. Install dependencies**

```bash
pip install torch torchvision pennylane scikit-learn matplotlib seaborn jupyter

```

## 🚀 How to Run

1. Launch the Jupyter environment:
```bash
jupyter notebook

```


2. Open the **"Quantum Neural Networks (QNNs) for Image Classification"** notebook file.
3. Ensure your Jupyter kernel is set to the `qnn_env` environment.
4. Run the cells sequentially to execute the dimensionality reduction, train the Hybrid QNN, evaluate the baseline Classical CNN, and generate comparative performance charts.

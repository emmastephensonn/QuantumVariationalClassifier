# Variational Quantum Classifier (VQC)

A Python implementation of a **Variational Quantum Classifier** built with Qiskit.  
This project demonstrates how hybrid quantum–classical machine learning models can perform binary classification using parameterized quantum circuits and classical optimization.

The model uses **angle encoding for classical data**, a **parameterized variational ansatz**, and a **classical optimizer** to train the circuit parameters.

---

# Overview

Variational quantum algorithms are hybrid models where:

1. Classical data is encoded into a quantum circuit
2. A parameterized ansatz applies trainable gates
3. Measurements are performed to generate predictions
4. A classical optimizer updates circuit parameters to minimize a loss function

This repository implements the full pipeline:

- Dataset generation and preprocessing
- Quantum circuit construction
- Hybrid quantum–classical training
- Model evaluation

---

## Project Structure
```
quantum_variational_classifier/
│
├── data/
│   └── dataset.py          # Dataset generation and preprocessing
│
├── circuits/
│   ├── encode.py           # Classical feature encoding
│   └── ansatz.py           # Parameterized quantum circuit
│
├── model/
│   └── quantum_model.py    # Circuit execution and measurement
│
├── training/
│   ├── cost.py             # Loss function
│   └── train.py            # Optimization loop
│
├── evaluation/
│   └── metrics.py          # Accuracy calculation
│
├── main.py                 # Runs the full experiment
├── requirements.txt        # Project dependencies
└── README.md               # Project documentation
```
---

# Model Architecture

## Data Encoding

Classical features are encoded using **angle encoding**:

Each feature `x_i` is mapped to a rotation gate:

`x_i → RY(x_i)`

Each feature controls a rotation applied to a corresponding qubit.

---

## Variational Ansatz

The ansatz consists of:

- parameterized single-qubit rotation gates
- entangling CNOT gates

These parameters are optimized during training to learn the classification boundary.

---

## Prediction

The expectation value of the Pauli-Z observable is used as the model output:

`<Z> ∈ [-1, 1]`

Predictions are mapped to class labels:
prediction >= 0 → class +1
prediction < 0 → class -1


---

## Loss Function

Training minimizes **Mean Squared Error (MSE)** between predictions and labels:

`L = (1/N) * Σ (y_pred - y_true)^2`

---

## Optimization

Circuit parameters are optimized using the **COBYLA optimizer** from SciPy.

---

# Dataset

The project uses a synthetic binary classification dataset generated with **scikit-learn's `make_moons`**.

Preprocessing steps include:

- scaling features to `[0, π]` for rotation encoding
- mapping labels `{0,1}` → `{−1,1}`
- splitting data into training and testing sets

---

# Performance Optimizations

Quantum circuit simulation can be computationally expensive.  
Several optimizations were implemented to improve training speed:

- reduced simulator shot count during training
- smaller dataset size
- limited optimizer iterations

Higher shot counts can be used during evaluation for more precise results.

---

# Installation

Clone the repository:
git clone https://github.com/emmastephensonn/QuantumVariationalClassifier.git

cd QuantumVariationalClassifier

Install dependencies:
pip install -r requirements.txt


Dependencies include:

- qiskit
- numpy
- scipy
- scikit-learn
- matplotlib

---

# Running the Project

Run the main script:
python main.py


The script will:

1. generate and preprocess the dataset  
2. train the variational quantum classifier  
3. evaluate model accuracy  
4. display the training loss curve  

Example output:
Test Accuracy: 0.84


---

# Example Results

Typical results from the model:

| Metric | Value |
|------|------|
| Accuracy | ~80–90% |
| Training Iterations | ~40 |
| Circuit Qubits | 2 |

Results may vary slightly due to quantum measurement noise.

---

# Future Improvements

Possible extensions include:

- deeper variational circuits
- alternative data encoding strategies
- SPSA optimizer for noisy quantum hardware
- running the model on real quantum hardware
- benchmarking against classical ML models

---

# Technologies Used

- Python
- Qiskit
- NumPy
- SciPy
- scikit-learn
- matplotlib

---

# Author

Emma Stephenson  
Nanotechnology Engineering – University of Waterloo

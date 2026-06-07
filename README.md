# Breast Cancer Classifier

A modular deep learning project for breast cancer classification using TensorFlow/Keras and the Breast Cancer Wisconsin dataset from `sklearn.datasets`.

The project follows clean ML engineering practices by separating:
- preprocessing
- model architecture
- training
- evaluation
- visualization
- utilities

into reusable Python modules.

---

# Project Structure

```text
breast_cancer_classifier/
│
├── data/
├── notebooks/
├── output/
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── visualizations.py
│   └── utils.py
│
├── main.py
└── README.md
```

---

# Dataset

The project uses:

```python
sklearn.datasets.load_breast_cancer()
```

Dataset contains:
- 30 numerical features
- binary classification
- malignant vs benign tumors

---

# Model Architecture

A classic Artificial Neural Network (ANN) built using TensorFlow/Keras.

Architecture:
- Dense Layer (15 neurons, ReLU)
- Dense Layer (10 neurons, ReLU)
- Output Layer (2 neurons, Softmax)

---

# Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- pandas
- matplotlib
- scikit-learn

---

# Features

- Modular ML project structure
- Data preprocessing pipeline
- Train/Test splitting
- Feature scaling using `StandardScaler`
- Neural network training
- Model evaluation
- Visualization utilities
- Reusable functions and clean architecture

---

# Training Results

Training Performance:

```text
accuracy: 0.9389
loss: 0.1690

validation accuracy: 0.8696
validation loss: 0.1814
```

Test Performance:

```text
Test Accuracy = 0.9473
```

The model achieved approximately **94.7% test accuracy** on the Breast Cancer Wisconsin dataset.

---

# How to Run the Project

## 1. Clone Repository

```bash
git clone <your-repo-link>
```

---

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 3. Run the Project

```bash
python main.py
```

---

# Example Workflow

The pipeline performs:
1. Data loading
2. Data preprocessing
3. Train/test split
4. Feature scaling
5. Model building
6. Model training
7. Evaluation
8. Visualization

---

# Future Improvements

Possible future enhancements:
- Add PyTorch implementation
- Hyperparameter tuning
- Model checkpointing
- FastAPI deployment
- Experiment tracking
- Docker support
- Advanced visualization dashboards

---

# Learning Goals

This project was built to practice:
- clean code principles
- modular ML architecture
- TensorFlow/Keras workflows
- software engineering practices for machine learning

---

# Author

Abdelrahman Yasser

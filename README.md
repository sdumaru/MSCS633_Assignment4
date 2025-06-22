# Credit Card Fraud Detection using AutoEncoder (PyOD)

This project demonstrates the use of a deep learning technique called AutoEncoder from the [PyOD library](https://pyod.readthedocs.io/en/latest/) to detect anomalies (fraudulent transactions) in a real-world anonymized credit card dataset.

---

## Dataset

- **Source:** [Kaggle - Fraud Detection Dataset](https://www.kaggle.com/datasets/whenamancodes/fraud-detection)
- The dataset contains numerical values of anonymized credit card transactions, including 30 features (`V1` to `V28`, `Amount`, and `Time`) and a binary `Class` label:
  - `0` – normal transaction
  - `1` – fraudulent transaction

---

## Technologies Used

- Python 3.x
- PyOD (AutoEncoder)
- Scikit-learn
- Pandas, NumPy
- Matplotlib, Seaborn
- Torch

---

##  Model Overview

An **AutoEncoder** neural network is trained only on normal transaction data to learn a compact representation. During inference, higher reconstruction error indicates a potential anomaly.

---

##  Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/sdumaru/MSCS633_Assignment4.git
cd MSCS633_Assignment4
```

### 2. Install Required Packages
```bash
pip install pandas numpy matplotlib seaborn scikit-learn pyod torch tensorflow
```

OR

```bash
pip install -r requirements.txt
```

### 3. Unzip the dataset (creditcard.zip)
Ensure creditcard.csv is in the same directory as the python script.

### 4. Run the Program
python .\fraud_detection.py

## References

- [AutoEncoder in PyOD](https://pyod.readthedocs.io/en/latest/pyod.models.html#module-pyod.models.auto_encoder)
- [Kaggle Dataset](https://www.kaggle.com/datasets/whenamancodes/fraud-detection)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html)
- [Sample AutoEncoder Notebook from Packt](https://github.com/PacktPublishing/Deep-Learning-and-XAI-Techniques-for-Anomaly-Detection/blob/main/Chapter1/PyOD_autoencoder/chapter1_pyod_autoencoder.ipynb)
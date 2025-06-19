#Reference: https://github.com/PacktPublishing/Deep-Learning-and-XAI-Techniques-for-Anomaly-Detection/blob/main/Chapter1/PyOD_autoencoder/chapter1_pyod_autoencoder.ipynb

import pandas as pd
import numpy as np
from pyod.models.auto_encoder import AutoEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("creditcard.csv")

# Optional: display first few rows
print(df.head())

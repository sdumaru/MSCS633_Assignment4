#Reference: https://github.com/PacktPublishing/Deep-Learning-and-XAI-Techniques-for-Anomaly-Detection/blob/main/Chapter1/PyOD_autoencoder/chapter1_pyod_autoencoder.ipynb

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pyod.models.auto_encoder import AutoEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Load dataset
df = pd.read_csv("creditcard.csv")

# Display first few rows
print(df.head())

# Data Preprocessing
X = df.drop(columns=["Class"])
y = df["Class"]

# Normalize feature data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# AutoEncoder model
clf = AutoEncoder(epoch_num=50, contamination=0.0017, hidden_neuron_list=[64, 32, 32, 64], verbose=1)
clf.fit(X_train)

# Prediction
y_test_pred = clf.predict(X_test)
y_test_scores = clf.decision_function(X_test)

# Evaluation
print("\nClassification Report:")
print(classification_report(y_test, y_test_pred, digits=4))

print("\n--- Confusion Matrix ---")
conf_matrix = confusion_matrix(y_test, y_test_pred)
print(conf_matrix)

# Plot 1: Visualization – Anomaly Scores (Auto Threshold)
plt.rcParams["figure.figsize"] = (15, 8)
plt.plot(y_test_scores)
plt.axhline(y=clf.threshold_, c='r', ls='dotted', label='threshold')
plt.xlabel('Instances')
plt.ylabel('Anomaly Scores')
plt.title('Anomaly Scores with Auto-Calculated Threshold')
plt.legend()
plt.show()

# Plot 2: Visualization – Anomaly Scores (Manual Threshold)
manual_threshold = 50
plt.plot(y_test_scores, color="green")
plt.axhline(y=manual_threshold, c='r', ls='dotted', label='threshold')
plt.xlabel('Instances')
plt.ylabel('Anomaly Scores')
plt.title('Anomaly Scores with Modified Threshold')
plt.legend()
plt.show()

# Plot 3: Scatterplot – Visualize transactions
df_subset = df.copy()
df_subset["Anomaly_Score"] = clf.decision_function(scaler.transform(df_subset.drop(columns=["Class"])))
plt.rcParams["figure.figsize"] = (15, 8)
sns.scatterplot(x="Time", y="Amount", hue="Anomaly_Score", data=df_subset, palette="RdBu_r", size="Anomaly_Score")
plt.xlabel('Time (seconds elapsed from first transaction)')
plt.ylabel('Transaction Amount')
plt.title('Transaction Time vs Amount Colored by Anomaly Score')
plt.legend(title='Anomaly Score', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
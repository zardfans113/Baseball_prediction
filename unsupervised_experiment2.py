import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ============ 📌 Step 1: Load and Preprocess Data ============
df = pd.read_csv("2024_qualified_hitters.csv")
df.columns = df.columns.str.replace(" ", "").str.strip()

features = ["打率", "本塁打", "打点", "出塁率", "長打率", "OPS"]
X = df[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ============ 📌 Step 2: PCA Dimensionality Reduction ============
pca = PCA(n_components=2)  # Reduce to 2 dimensions for visualization
X_pca = pca.fit_transform(X_scaled)

# Apply K-Means clustering on PCA-transformed data
kmeans_pca = KMeans(n_clusters=3, random_state=42, n_init=100)
y_pred_pca = kmeans_pca.fit_predict(X_pca)

# Compute Silhouette Score for PCA
silhouette_pca = silhouette_score(X_pca, y_pred_pca)

print("\n✅ PCA + K-Means Clustering Result")
print(f"🔹 PCA Silhouette Score: {silhouette_pca:.4f}")

# ============ 📌 Step 3: Autoencoder for Dimensionality Reduction ============
encoding_dim = 2  # Reduce to 2 dimensions for comparison
input_dim = X_scaled.shape[1]

# Define Autoencoder architecture
input_layer = Input(shape=(input_dim,))
encoded = Dense(10, activation='relu')(input_layer)
encoded = Dense(encoding_dim, activation='relu')(encoded)  # Latent space (2D)
decoded = Dense(10, activation='relu')(encoded)
decoded = Dense(input_dim, activation='sigmoid')(decoded)

autoencoder = Model(input_layer, decoded)
encoder = Model(input_layer, encoded)

# Compile and train Autoencoder
autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.fit(X_scaled, X_scaled, epochs=200, batch_size=16, verbose=0)

# Encode data using the trained encoder
X_encoded = encoder.predict(X_scaled)

# Apply K-Means clustering on Autoencoder-transformed data
kmeans_autoencoder = KMeans(n_clusters=3, random_state=42, n_init=100)
y_pred_autoencoder = kmeans_autoencoder.fit_predict(X_encoded)

# Compute Silhouette Score for Autoencoder
silhouette_autoencoder = silhouette_score(X_encoded, y_pred_autoencoder)

print("\n✅ Autoencoder + K-Means Clustering Result")
print(f"🔹 Autoencoder Silhouette Score: {silhouette_autoencoder:.4f}")

# ============ 📌 Step 4: Visualization ============
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# PCA Clustering Plot
axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=y_pred_pca, cmap="viridis", alpha=0.7)
axes[0].set_xlabel("PCA Component 1")
axes[0].set_ylabel("PCA Component 2")
axes[0].set_title(f"PCA + K-Means Clustering (Silhouette Score: {silhouette_pca:.4f})")

# Autoencoder Clustering Plot
axes[1].scatter(X_encoded[:, 0], X_encoded[:, 1], c=y_pred_autoencoder, cmap="viridis", alpha=0.7)
axes[1].set_xlabel("Encoded Feature 1")
axes[1].set_ylabel("Encoded Feature 2")
axes[1].set_title(f"Autoencoder + K-Means Clustering (Silhouette Score: {silhouette_autoencoder:.4f})")

plt.tight_layout()
plt.show()

# ============ 📌 Step 5: Conclusion ============
if silhouette_autoencoder > silhouette_pca:
    print("\n✅ Conclusion: Autoencoder performed better for clustering!")
else:
    print("\n✅ Conclusion: PCA performed better for clustering!")

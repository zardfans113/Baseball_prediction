import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ============ 📌 Step 1: Load and Preprocess Data ============
df = pd.read_csv("2024_qualified_hitters.csv")

# Clean column names (remove spaces)
df.columns = df.columns.str.replace(" ", "").str.strip()

# Select relevant features for clustering
features = ["打率", "本塁打", "打点", "出塁率", "長打率", "OPS"]
X = df[features]

# Standardize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ============ 📌 Step 2: Build Autoencoder for Feature Extraction ============
encoding_dim = 5  # Dimensionality reduction target
input_dim = X_scaled.shape[1]

# Define encoder-decoder structure
input_layer = Input(shape=(input_dim,))
encoded = Dense(10, activation='relu')(input_layer)
encoded = Dense(encoding_dim, activation='relu')(encoded)  # Latent space representation
decoded = Dense(10, activation='relu')(encoded)
decoded = Dense(input_dim, activation='sigmoid')(decoded)

# Compile Autoencoder
autoencoder = Model(input_layer, decoded)
encoder = Model(input_layer, encoded)

autoencoder.compile(optimizer='adam', loss='mse')

# Train Autoencoder
autoencoder.fit(X_scaled, X_scaled, epochs=200, batch_size=16, verbose=0)

# Extract compressed feature representation
X_encoded = encoder.predict(X_scaled)

# ============ 📌 Step 3: Apply K-Means Clustering ============
kmeans = KMeans(n_clusters=3, random_state=42, n_init=100)
y_pred_init = kmeans.fit_predict(X_encoded)

# Compute Silhouette Score
silhouette_initial = silhouette_score(X_encoded, y_pred_init)

print("\n✅ DEC + K-Means Initial Clustering Result")
print(f"🔹 Initial Silhouette Score: {silhouette_initial:.4f}")

# ============ 📌 Step 4: Data Augmentation with Gaussian Noise ============
# Add 5% random Gaussian noise
X_augmented = X_scaled + np.random.normal(0, 0.05, X_scaled.shape)

# Train Autoencoder again with noisy data
autoencoder.fit(X_augmented, X_augmented, epochs=200, batch_size=16, verbose=0)

# Encode noisy data
X_encoded_augmented = encoder.predict(X_augmented)

# K-Means Clustering on Augmented Data
kmeans_augmented = KMeans(n_clusters=3, random_state=42, n_init=100)
y_pred_augmented = kmeans_augmented.fit_predict(X_encoded_augmented)

# Compute new Silhouette Score
silhouette_augmented = silhouette_score(X_encoded_augmented, y_pred_augmented)

print("\n✅ DEC + K-Means with Data Augmentation")
print(f"🔹 Silhouette Score After Data Augmentation: {silhouette_augmented:.4f}")

# ============ 📌 Step 5: Visualization ============
plt.figure(figsize=(12, 6))

# Plot original clustering
plt.subplot(1, 2, 1)
plt.scatter(X_encoded[:, 0], X_encoded[:, 1], c=y_pred_init, cmap="viridis", alpha=0.7)
plt.xlabel("Encoded Feature 1")
plt.ylabel("Encoded Feature 2")
plt.title(f"Original Clustering (Silhouette: {silhouette_initial:.4f})")
plt.colorbar(label="Cluster ID")

# Plot clustering after data augmentation
plt.subplot(1, 2, 2)
plt.scatter(X_encoded_augmented[:, 0], X_encoded_augmented[:, 1], c=y_pred_augmented, cmap="viridis", alpha=0.7)
plt.xlabel("Encoded Feature 1")
plt.ylabel("Encoded Feature 2")
plt.title(f"With Data Augmentation (Silhouette: {silhouette_augmented:.4f})")
plt.colorbar(label="Cluster ID")

plt.tight_layout()
plt.show()

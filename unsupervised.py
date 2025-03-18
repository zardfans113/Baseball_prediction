import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import pandas as pd

# 設定日文字體（使用 MS Gothic、Yu Gothic、Meiryo 或 Noto Sans CJK JP）
plt.rcParams['font.sans-serif'] = ['MS Gothic']  # 或 'Meiryo', 'Yu Gothic', 'Noto Sans CJK JP'
plt.rcParams['axes.unicode_minus'] = False  # 避免負號顯示錯誤
# ============ 📌 Step 1: 讀取並整理數據 ============
df = pd.read_csv("2024_qualified_hitters.csv")
df.columns = df.columns.str.replace(" ", "").str.strip()

features = ["打率", "本塁打", "打点", "出塁率", "長打率", "OPS"]
X = df[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ============ 📌 Step 2: 構建 Autoencoder ============
encoding_dim = 5  # 降維
input_dim = X_scaled.shape[1]

input_layer = Input(shape=(input_dim,))
encoded = Dense(10, activation='relu')(input_layer)
encoded = Dense(encoding_dim, activation='relu')(encoded)  # Latent space
decoded = Dense(10, activation='relu')(encoded)
decoded = Dense(input_dim, activation='sigmoid')(decoded)

autoencoder = Model(input_layer, decoded)
encoder = Model(input_layer, encoded)

autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.fit(X_scaled, X_scaled, epochs=200, batch_size=16, verbose=0)

X_encoded = encoder.predict(X_scaled)

# ============ 📌 Step 3: Deep Embedded Clustering (DEC) ============
kmeans = KMeans(n_clusters=3, random_state=42, n_init=100)
y_pred_init = kmeans.fit_predict(X_encoded)

# 計算 K-Means 評估指標
silhouette_avg = silhouette_score(X_encoded, y_pred_init)

print("\n✅ DEC + K-Means 初始分群結果")
print(f"🔹 Silhouette Score: {silhouette_avg:.4f}")

# ============ 📌 Step 4: 可視化 ============
plt.figure(figsize=(8, 6))
plt.scatter(X_encoded[:, 0], X_encoded[:, 1], c=y_pred_init, cmap="viridis", alpha=0.7)
plt.xlabel("Encoded Feature 1")
plt.ylabel("Encoded Feature 2")
plt.title("Deep Embedded Clustering 分群結果")
plt.colorbar(label="Cluster ID")
plt.show()

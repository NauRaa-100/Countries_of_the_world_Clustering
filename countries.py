"""
Countries of the World Clustering
"""

# ===========================
# Import Libraries
# ===========================
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering, MiniBatchKMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# ===========================
# Load Dataset
# ===========================
df_ = pd.read_csv(r'E:\Rev-DataScience\AI-ML\countries.csv', encoding='latin')

print(df_.shape)
print("Missing values:", df_.isnull().sum().sum())

# ===========================
# Memory Optimization + Missing Values
# ===========================
for col in df_.columns:
    if df_[col].dtype == 'int64':
        df_[col] = df_[col].astype('int16').fillna(df_[col].median())
    elif df_[col].dtype == 'float64':
        df_[col] = df_[col].astype('float32').fillna(df_[col].median())
    else:
        df_[col] = df_[col].fillna(df_[col].mode()[0])

print("Duplicated rows:", df_.duplicated().sum())

# ===========================
# Encoding
# ===========================
df = pd.get_dummies(df_, drop_first=True)

joblib.dump(df.columns.to_list(), "all_features.joblib")

# ===========================
# Outlier Removal
# ===========================
iso = IsolationForest(contamination=0.02, random_state=42)
mask = iso.fit_predict(df) != -1
df = df[mask]

# ===========================
# Feature Selection
# ===========================
x = df.values
selector = VarianceThreshold(threshold=0.1)
x_reduced = selector.fit_transform(x)
mask = selector.get_support()
selected_features = df.columns[mask].tolist()
print(f" Selected {len(selected_features)} features after VarianceThreshold")

# ===========================
# Scaling
# ===========================
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x_reduced)

# ===========================
# PCA
# ===========================
n_components = min(10, x_scaled.shape[1]) 
pca = PCA(n_components=n_components, random_state=42)
x_pca = pca.fit_transform(x_scaled)

plt.figure(figsize=(7, 5))
plt.scatter(x_pca[:, 0], x_pca[:, 1], alpha=0.7, s=40)
plt.title("PCA Projection Before Clustering")
plt.show()

# ===========================
# EDA & Visualization
# ===========================
importances = pd.Series(selector.variances_, index=df.columns)
features = importances[mask].sort_values(ascending=False).head(5).index.tolist()

for col in df[features]:
    if df[col].skew() > 0.5:
        plt.figure(figsize=(9, 6))
        sns.histplot(df[col], kde=True, color='blue')
        plt.title(f"{col} Before Log Transform")
        plt.show()

        df[col] = np.log1p(df[col])

        plt.figure(figsize=(9, 6))
        sns.histplot(df[col], kde=True, color='red')
        plt.title(f"{col} After Log Transform")
        plt.show()

corr = df[features].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='Blues')
plt.title("Correlation Of Top Features")
plt.show()

# ===========================
# Model Benchmarking
# ===========================
models = {
    "KMeans": KMeans(n_clusters=7, random_state=42),
    "Agglomerative": AgglomerativeClustering(n_clusters=7, linkage='ward'),
    "DBSCAN": DBSCAN(eps=0.7, min_samples=5),
    "Spectral": SpectralClustering(n_clusters=7, random_state=42, affinity='nearest_neighbors'),
    "GaussianMixture": GaussianMixture(n_components=7, random_state=42),
    "MiniBatchKMeans": MiniBatchKMeans(n_clusters=7, random_state=42)
}

scores = {}
for name, m in models.items():
    try:
        labels = m.fit_predict(x_pca)
        if len(set(labels)) > 1:
            score = silhouette_score(x_pca, labels)
            scores[name] = score
            print(f"{name}: silhouette = {score:.3f}")
    except Exception as e:
        print(f"{name} failed: {e}")

# ===========================
# Pick the Best Model
# ===========================

model = GaussianMixture(n_components=7, random_state=42)
labels = model.fit_predict(x_pca)

df["cluster"] = labels

plt.figure(figsize=(8, 5))
sns.scatterplot(x=x_pca[:, 0], y=x_pca[:, 1], hue=labels, palette="coolwarm")
plt.title(f"Clusters Visualization ({model})")
plt.show()

# ===========================
# Save Everything
# ===========================
joblib.dump(model, "best_cluster_model.joblib")
joblib.dump(pca, "pca_transform.joblib")
joblib.dump(scaler, "scaler.joblib")
joblib.dump(selector, "selector.joblib")

print(" Done — Model and transformers saved successfully!")

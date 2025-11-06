import gradio as gr
import pandas as pd
import numpy as np
import joblib
import traceback
import matplotlib.pyplot as plt

# ===========================
# Load Model & Transformers
# ===========================
model = joblib.load("best_cluster_model.joblib")
pca = joblib.load("pca_transform.joblib")
scaler = joblib.load("scaler.joblib")
selector = joblib.load("selector.joblib")
selected_features = joblib.load("all_features.joblib")

# ===========================
# Prediction Function
# ===========================
def predict_clusters(file):
    try:
        df_new = pd.read_csv(file.name)
    except Exception as e:
        return f"❌ Error loading file: {e}", None

    # Fill missing values
    for col in df_new.columns:
        if df_new[col].dtype in ["float64", "float32", "int64", "int32","int8"]:
            df_new[col] = df_new[col].fillna(df_new[col].median())
        else:
            df_new[col] = df_new[col].fillna(df_new[col].mode()[0])

    df_new = pd.get_dummies(df_new, drop_first=True)

    for col in selected_features:
        if col not in df_new.columns:
            df_new[col] = 0

    df_new = df_new[selected_features]

    try:
        X_selected = selector.transform(df_new)
        X_scaled = scaler.transform(X_selected)
        X_pca = pca.transform(X_scaled)

        # Predict clusters
        labels = model.predict(X_pca)
        df_out = pd.DataFrame({
            "PC1": X_pca[:, 0],
            "PC2": X_pca[:, 1],
            "Cluster": labels
        })

        # ========================
        # Create Cluster Plot
        # ========================
        fig, ax = plt.subplots(figsize=(7,5))
        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', s=50)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title("PCA Projection + Cluster Assignment")
        legend1 = ax.legend(*scatter.legend_elements(), title="Cluster")
        ax.add_artist(legend1)
        plt.tight_layout()

        return df_out, fig

    except Exception as e:
        tb = traceback.format_exc()
        return f"❌ Error during transformation or prediction:\n{e}\n\n{tb}", None

# ===========================
# Gradio Interface
# ===========================
interface = gr.Interface(
    fn=predict_clusters,
    inputs=gr.File(label="📂 Upload your CSV dataset"),
    outputs=[
        gr.Dataframe(label="🧭 Clustered Results (PCA + Cluster)"),
        gr.Plot(label="📊 Cluster Visualization")
    ],
    title="🌍 Countries of the World Clustering App",
    description=(
        "Upload your dataset to predict country clusters using the pre-trained unsupervised model "
        "(VarianceThreshold + StandardScaler + PCA + GaussianMixture)."
    )
)

if __name__ == "__main__":
    interface.launch()

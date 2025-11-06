
# Countries of the World Clustering

This repository contains a complete **unsupervised clustering workflow** for countries based on various attributes. The app is also deployed on Hugging Face Spaces:

🔗 [Hugging Face Space - Countries Clustering](https://huggingface.co/spaces/NauRaa/Countries_of_the_world_Clustering_App)

---

##  Features

- **Data Cleaning & Memory Optimization**  
  Handles missing values, optimizes memory usage, and removes duplicates.

- **Encoding & Feature Selection**  
  One-hot encoding for categorical variables and VarianceThreshold for feature selection.

- **Outlier Detection**  
  Uses `IsolationForest` to detect and remove outliers automatically.

- **Scaling & Dimensionality Reduction**  
  StandardScaler + PCA (up to 10 components) for transforming data before clustering.

- **Model Benchmarking**  
  Compares multiple clustering models including:
  - KMeans
  - AgglomerativeClustering
  - DBSCAN
  - SpectralClustering
  - GaussianMixture
  - MiniBatchKMeans

- **Visualization**  
  - PCA scatter plots  
  - Feature distributions before and after log-transform  
  - Correlation heatmaps  
  - Cluster visualizations for the selected model

- **Deployment Ready**  
  The trained model, scaler, selector, and PCA transformer are saved for deployment in a Gradio/Hugging Face Space.

---

##  Installation

```bash
# Clone the repository
git clone https://github.com/NauRaa-100/Countries_of_the_world_Clustering/blob/main/countries.py

# Install dependencies
pip install -r requirements.txt


---

##  How to Use

1. Place your dataset (`countries.csv`) in the project folder.
2. Run the main script to train and save the model:

```bash
python main.py
```

3. Deploy the app locally with Gradio or directly use the Hugging Face Space.

4. Upload your CSV file in the app interface to predict clusters and visualize results.

---

##  Saved Artifacts

* `best_cluster_model.joblib` — The trained clustering model (`GaussianMixture` by default)
* `pca_transform.joblib` — PCA transformer
* `scaler.joblib` — StandardScaler
* `selector.joblib` — VarianceThreshold selector
* `all_features.joblib` — Original feature list

These files are loaded by the Gradio/Hugging Face app for real-time predictions.

---

##  Notes

* PCA is used for dimensionality reduction, and log transformation is applied to skewed features.
* The app is fully functional on Hugging Face Spaces: users can upload any compatible dataset and get **cluster assignments + PCA visualizations** instantly.
* GaussianMixture is used as the default model for prediction, but other models were benchmarked and can be switched if needed.

---

##  License

This project is licensed under the MIT License.

---

Enjoy clustering countries around the world! 

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Suppose 'embeddings' is your numpy array of shape (10, 384)
# For demonstration, let's create random data
embeddings = np.random.rand(40, 384)  # Replace this with your actual embeddings
# If you already have your embeddings from your model:
# e.g., embeddings = model_output_mu  (or whichever output you want to visualize)

# Use t-SNE to reduce dimensions to 2
tsne = TSNE(n_components=2, random_state=42)
embeddings_2d = tsne.fit_transform(embeddings)

plt.figure(figsize=(8, 6))
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], s=100, alpha=0.7)
for i, (x, y) in enumerate(embeddings_2d):
    plt.text(x, y, str(i), fontsize=12)  # Label each point by its sample index
plt.title("t-SNE Visualization of Node Embeddings")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.show()

from sklearn.decomposition import PCA

pca = PCA(n_components=2)
embeddings_2d = pca.fit_transform(embeddings)

plt.figure(figsize=(8, 6))
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], s=100, alpha=0.7)
for i, (x, y) in enumerate(embeddings_2d):
    plt.text(x, y, str(i), fontsize=12)
plt.title("PCA Visualization of Node Embeddings")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.show()

import seaborn as sns
from sklearn.metrics.pairwise import euclidean_distances

# Compute the pairwise Euclidean distance matrix
distance_matrix = euclidean_distances(embeddings)

plt.figure(figsize=(8, 6))
sns.heatmap(distance_matrix, annot=True, fmt=".2f", cmap="viridis")
plt.title("Pairwise Euclidean Distance Between Embeddings")
plt.xlabel("Sample Index")
plt.ylabel("Sample Index")
plt.show()

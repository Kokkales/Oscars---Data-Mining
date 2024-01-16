import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn.metrics import confusion_matrix

TRAIN_PATH_PROCESSED = "./Data/moviesUpdated_processed.xlsx"
FULL_PREDICTIONS = './Data/full_predictions.xlsx'

def separate_data(ds):
    if 'predictions' not in ds.columns:
        raise ValueError('predictions not in the dataset')
    target = ds['predictions']
    data = ds.drop(columns='predictions')
    data = data.sort_index(axis=1)
    return target, data

iris = pd.read_excel(FULL_PREDICTIONS, sheet_name='Sheet1')
y, X = separate_data(iris)

# Apply PCA to reduce dimensions to 2 for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Perform spectral clustering
num_clusters = 4
spectral_clu = SpectralClustering(n_clusters=num_clusters, affinity='nearest_neighbors', n_neighbors=10)  # You can adjust parameters
spectral_labels = spectral_clu.fit_predict(X_pca)

# Evaluate clustering performance
print(confusion_matrix(y, spectral_labels))
print(metrics.calinski_harabasz_score(X_pca, spectral_labels))
print(metrics.silhouette_score(X_pca, spectral_labels, metric='euclidean'))

# Plot the spectral clusters
for i in range(num_clusters):
    cluster_df = pd.DataFrame(X_pca[spectral_labels == i], columns=['PC1', 'PC2'])
    plt.scatter(cluster_df['PC1'], cluster_df['PC2'], label=f'Cluster {i + 1}')

plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title(f'{num_clusters} Clusters Spectral Clustering')
plt.legend()
plt.show()

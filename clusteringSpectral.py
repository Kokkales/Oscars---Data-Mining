import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn.metrics import confusion_matrix

TRAIN_PATH_PROCESSED = "./Data/moviesUpdated_processed.xlsx"
FULL_PREDICTIONS = './Data/full_predictions.xlsx'

def get_rows_with_prediction_one(df):
    if 'predictions' not in df.columns:
        raise ValueError('predictions not in the dataframe')

    prediction_one_ids = (df[df['predictions'] == 1].index).tolist()
    return prediction_one_ids

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

# Identify points with prediction equal to 1
prediction_one_ids = get_rows_with_prediction_one(iris)

# Add cluster labels and prediction_one_ids to your dataframe
dfcluster_spectral = pd.DataFrame(spectral_labels, columns=['spectral_cluster'])
dfpca = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
dfclass = pd.DataFrame(y, columns=['predictions'])
dfall_spectral = pd.concat([dfpca, dfclass, dfcluster_spectral], axis=1)

# Print clustering stats
print("Confusion Matrix:")
print(confusion_matrix(dfall_spectral['predictions'], dfall_spectral['spectral_cluster']))
print("Calinski-Harabasz Score:", metrics.calinski_harabasz_score(X_pca, dfall_spectral['spectral_cluster']))
print("Silhouette Score:", metrics.silhouette_score(X_pca, dfall_spectral['spectral_cluster'], metric='euclidean'))

# Visualize clusters in the reduced-dimensional space
for i in range(num_clusters):
    cluster_df = dfall_spectral[dfall_spectral['spectral_cluster'] == i]
    plt.scatter(cluster_df['PC1'], cluster_df['PC2'], label=f'Cluster {i + 1}')

# Print points with prediction equal to 1
for index, row in dfall_spectral.iterrows():
    point_id = index
    pc1_value = row['PC1']
    pc2_value = row['PC2']
    cluster_label = row['spectral_cluster']

    if point_id in prediction_one_ids:
        print(f"Point ID: {point_id+1}, PC1: {pc1_value}, PC2: {pc2_value}, Cluster: {cluster_label}")

plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title(f'{num_clusters} Clusters Spectral Clustering')
plt.legend()
plt.show()

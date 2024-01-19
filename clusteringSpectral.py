import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from sklearn.cluster import SpectralClustering
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler

TRAIN_PATH_PROCESSED = "./Data/moviesUpdated_processed.xlsx"
FULL_PREDICTIONS = './Data/full_predictions.xlsx'

def separateData(ds):
    if 'oscar winners' not in ds.columns:
        raise ValueError('oscar winners not in the dataset')
    target = ds['oscar winners']
    data = ds.drop(columns='oscar winners')
    # data = data.sort_index(axis=1)
    return target, data

def get_rows_with_prediction_one(df):
    if 'oscar winners' not in df.columns:
        raise ValueError('oscar winners not in the dataframe')

    prediction_one_ids = (df[df['oscar winners'] == 1].index).tolist()
    return prediction_one_ids

if len(sys.argv)!=3:
    print("Usage: python3 clustering.py <algorithm> <cluster number> <ss/rs/mm>")
    sys.exit(1)

iris = pd.read_excel(TRAIN_PATH_PROCESSED, sheet_name='Sheet1')
y, X = separateData(iris)
prediction_one_ids = get_rows_with_prediction_one(iris)

if sys.argv[2]=='ss':
    scaler = StandardScaler()
elif sys.argv[2]=='rs':
    scaler=RobustScaler()
else:
    scaler=MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Perform PCA on the scaled data
pca = PCA(n_components=2)
Xnew = pca.fit_transform(X_scaled)
feature_names_pc1 = X.columns[np.argsort(pca.components_[0])[::-1]].tolist()

# Perform Spectral Clustering on the scaled data
num_clusters = int(sys.argv[1])  # You can choose the number of clusters
spectral_clustering = SpectralClustering(n_clusters=num_clusters, affinity='nearest_neighbors', random_state=42)
spectral_labels = spectral_clustering.fit_predict(Xnew)

# Add cluster labels to your dataframe
dfcluster_spectral = pd.DataFrame(spectral_labels, columns=['spectral_cluster'])
dfpca = pd.DataFrame(Xnew, columns=["PC1", "PC2"])
dfclass = pd.DataFrame(y, columns=['oscar winners'])
dfall_spectral = pd.concat([dfpca, dfclass, dfcluster_spectral], axis=1)

# Print clustering stats
print("Confusion Matrix:")
print(confusion_matrix(dfall_spectral['oscar winners'], dfall_spectral['spectral_cluster']))
print("Silhouette Score:", silhouette_score(Xnew, dfall_spectral['spectral_cluster'], metric='euclidean'))

# Explore cluster characteristics (centroid values) - Spectral Clustering doesn't have explicit centroids

# Loop through the Spectral Clusters and print information for points with prediction equal to 1
pc1_osc=[]
pc2_osc=[]
for index, row in dfall_spectral.iterrows():
    point_id = index  # Assuming the index is the ID, you can adjust this based on your dataset
    pc1_value = row['PC1']
    pc2_value = row['PC2']
    cluster_label = row['spectral_cluster'] + 1
    if point_id in prediction_one_ids:
        pc1_osc.append(pc1_value)
        pc2_osc.append(pc2_value)
        # print(f"Point ID: {point_id+1}, PC1: {pc1_value}, PC2: {pc2_value}, Cluster: {cluster_label}")

# Plot clusters according to principal components 1 and 2
fig, axs = plt.subplots(1, 2, figsize=(10, 5))

for i in range(num_clusters):
    cluster_df = dfall_spectral[dfall_spectral['spectral_cluster'] == i]
    axs[0].scatter(cluster_df['PC1'], cluster_df['PC2'], label=f'Cluster {i + 1}')
axs[0].scatter(pc1_osc, pc2_osc, label=f'Oscars',color='red')
axs[0].set_xlabel("PC1")
axs[0].set_ylabel("PC2")
axs[0].set_title("Spectral Clusters in PC1-PC2 Space")
axs[0].legend()

# Plot the silhouette score
sils = []
for i in range(2, 11):
    spectral_clustering = SpectralClustering(n_clusters=i, affinity='nearest_neighbors', random_state=42)
    spectral_labels = spectral_clustering.fit_predict(Xnew)
    sils.append(silhouette_score(Xnew, spectral_labels, metric='euclidean'))
axs[1].plot(range(2, 11), sils, marker='o')
axs[1].set_xlabel('Number of Clusters')
axs[1].set_ylabel('Silhouette Score')
axs[1].set_title('Silhouette Score')

plt.tight_layout()
plt.show()
plt.close()

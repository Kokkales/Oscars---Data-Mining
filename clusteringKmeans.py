import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from sklearn.cluster import KMeans
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

# Perform K-Means clustering on the scaled data
num_clusters = int(sys.argv[1])  # You can choose the number of clusters
kclu = KMeans(n_clusters=num_clusters, n_init='auto', random_state=42)
kclu.fit(Xnew)

# Add cluster labels to your dataframe
dfcluster_kmeans = pd.DataFrame(kclu.labels_, columns=['kmeans_cluster'])
dfpca = pd.DataFrame(Xnew, columns=["PC1", "PC2"])
dfclass = pd.DataFrame(y, columns=['oscar winners'])
dfall_kmeans = pd.concat([dfpca, dfclass, dfcluster_kmeans], axis=1)

# Print clustering stats
print("Confusion Matrix:")
print(confusion_matrix(dfall_kmeans['oscar winners'], dfall_kmeans['kmeans_cluster']))
print("Calinski-Harabasz Score:", metrics.calinski_harabasz_score(Xnew, dfall_kmeans['kmeans_cluster']))
print("Silhouette Score:", metrics.silhouette_score(Xnew, dfall_kmeans['kmeans_cluster'], metric='euclidean'))

# Explore cluster characteristics (centroid values)
centroids = kclu.cluster_centers_
for i in range(num_clusters):
    print(f"Cluster {i + 1} Centroid Values:")
    print("PC1:", centroids[i, 0])
    print("PC2:", centroids[i, 1])
    print()

feature_importance_pc1 = pca.components_[0]
feature_importance_pc2 = pca.components_[1]

print("Top features contributing to PC1:")
top_features_pc1 = X.columns[np.argsort(feature_importance_pc1)[::-1]][:5].tolist()
print(top_features_pc1)

print("\nTop features contributing to PC2:")
top_features_pc2 = X.columns[np.argsort(feature_importance_pc2)[::-1]][:5].tolist()
print(top_features_pc2)

# Loop through the K-Means clusters and print information for points with prediction equal to 1
pc1_osc=[]
pc2_osc=[]
for index, row in dfall_kmeans.iterrows():
    point_id = index  # Assuming the index is the ID, you can adjust this based on your dataset
    pc1_value = row['PC1']
    pc2_value = row['PC2']
    cluster_label = row['kmeans_cluster'] + 1
    if point_id in prediction_one_ids:
        pc1_osc.append(pc1_value)
        pc2_osc.append(pc2_value)
        # print(f"Point ID: {point_id+1}, PC1: {pc1_value}, PC2: {pc2_value}, Cluster: {cluster_label}")

# Explore cluster characteristics (centroid values)
centroids = kclu.cluster_centers_
# Display top features contributing to each cluster
print("Top features contributing to each cluster:")
for i in range(num_clusters):
    print(f"Cluster {i + 1}:")
    # Get the indices of the top features for the current cluster
    top_features_indices = np.argsort(centroids[i])[::-1][:7]
    top_features = X.columns[top_features_indices].tolist()
    print(top_features)

fig, axs = plt.subplots(1, 3, figsize=(15, 5))
# Plot the silhouette score comparing to Fowlkes-Mallows score
xs = range(2, 50)
sils = []
fms = []

for i in xs:
    kclu = KMeans(n_clusters=i, n_init='auto')
    kclu.fit(Xnew)
    sils.append(metrics.silhouette_score(Xnew, kclu.labels_, metric='euclidean'))
    fms.append(metrics.fowlkes_mallows_score(dfall_kmeans['oscar winners'], kclu.labels_))
axs[0].plot(xs, sils)
axs[0].plot(xs, fms)
axs[0].set_xlabel('Number of clusters (k)')
axs[0].set_ylabel('Score')
axs[0].set_title('Silhouette and Fowlkes-Mallows Scores')
axs[0].legend(['Silhouette', 'Fowlkes-Mallows'])

# Plot silhouette scores
silhouette_scores = []
for i in range(2, 11):
    kmeans = KMeans(n_clusters=i, n_init='auto',  random_state=42)
    kmeans.fit(Xnew)
    silhouette_scores.append(metrics.silhouette_score(Xnew, kmeans.labels_))
axs[1].plot(range(2, 11), silhouette_scores, marker='o')
axs[1].set_xlabel('Number of Clusters')
axs[1].set_ylabel('Silhouette Score')
axs[1].set_title('Silhouette Score')

# Plot clusters according to principa components 1 and 2
for i in range(num_clusters):
    cluster_df = dfall_kmeans[dfall_kmeans['kmeans_cluster'] == i]
    axs[2].scatter(cluster_df['PC1'], cluster_df['PC2'], label=f'Cluster {i + 1}')
axs[2].scatter(pc1_osc, pc2_osc, label=f'Oscars',color='red')
axs[2].set_xlabel("PC1")
axs[2].set_ylabel("PC2")
axs[2].set_title("K-Means Clusters in PC1-PC2 Space")
axs[2].legend()

plt.tight_layout()
plt.show()
plt.close()

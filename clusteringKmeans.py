# # ----------------------------------------------------------------------------------TEST 1
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import sys
# from sklearn.cluster import KMeans
# from sklearn.decomposition import PCA
# from sklearn import metrics
# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import silhouette_score

# TRAIN_PATH_PROCESSED = "./Data/moviesUpdated_processed.xlsx"
# FULL_PREDICTIONS = './Data/full_predictions.xlsx'

# def separateData(ds):
#     if 'oscar winners' not in ds.columns:
#         raise ValueError('oscar winners not in the dataset')
#     target = ds['oscar winners']
#     data = ds.drop(columns='oscar winners')
#     # data = data.sort_index(axis=1)
#     return target, data

# def get_rows_with_prediction_one(df):
#     if 'oscar winners' not in df.columns:
#         raise ValueError('oscar winners not in the dataframe')

#     prediction_one_ids = (df[df['oscar winners'] == 1].index).tolist()
#     return prediction_one_ids

# iris = pd.read_excel(TRAIN_PATH_PROCESSED, sheet_name='Sheet1')
# y, X = separateData(iris)



# # Example usage:
# # iris = pd.read_excel(FULL_PREDICTIONS, sheet_name='Sheet1')
# prediction_one_ids = get_rows_with_prediction_one(iris)

# print("IDs of rows with prediction equal to 1:")
# print(prediction_one_ids)

# # Perform PCA
# pca = PCA(n_components=2)
# Xnew = pca.fit_transform(X)
# feature_names_pc1 = X.columns[np.argsort(pca.components_[0])[::-1]].tolist()

# # Perform K-Means clustering
# num_clusters = int(sys.argv[1])# You can choose the number of clusters
# kclu = KMeans(n_clusters=num_clusters, n_init=50, random_state=42)
# kclu.fit(Xnew)
# # hac = AgglomerativeClustering(n_clusters=num_clusters)
# # hac_labels = hac.fit_predict(Xnew)

# # Add cluster labels to your dataframe
# dfcluster_kmeans = pd.DataFrame(kclu.labels_, columns=['kmeans_cluster'])
# dfpca = pd.DataFrame(Xnew, columns=["PC1", "PC2"])
# dfclass = pd.DataFrame(y, columns=['oscar winners'])
# dfall_kmeans = pd.concat([dfpca, dfclass, dfcluster_kmeans], axis=1)

# # Print clustering stats
# print("Confusion Matrix:")
# print(confusion_matrix(dfall_kmeans['oscar winners'], dfall_kmeans['kmeans_cluster']))
# print("Calinski-Harabasz Score:", metrics.calinski_harabasz_score(Xnew, dfall_kmeans['kmeans_cluster']))
# print("Silhouette Score:", metrics.silhouette_score(Xnew, dfall_kmeans['kmeans_cluster'], metric='euclidean'))


# # Explore cluster characteristics (centroid values)
# centroids = kclu.cluster_centers_
# for i in range(num_clusters):
#     print(f"Cluster {i + 1} Centroid Values:")
#     print("PC1:", centroids[i, 0])
#     print("PC2:", centroids[i, 1])
#     print()
# feature_importance_pc1 = pca.components_[0]
# feature_importance_pc2 = pca.components_[1]

# print("Top features contributing to PC1:")
# top_features_pc1 = X.columns[np.argsort(feature_importance_pc1)[::-1]][:5].tolist()
# print(top_features_pc1)

# print("\nTop features contributing to PC2:")
# top_features_pc2 = X.columns[np.argsort(feature_importance_pc2)[::-1]][:5].tolist()
# print(top_features_pc2)
# # print(0['kmeans_cluster'])
# for index, row in dfall_kmeans.iterrows():
#     # print(row)
#     point_id = index  # Assuming the index is the ID, you can adjust this based on your dataset
#     pc1_value = row['PC1']
#     pc2_value = row['PC2']
#     cluster_label = row['kmeans_cluster']+1
#     if point_id in prediction_one_ids:
#       print(f"Point ID: {point_id+1}, PC1: {pc1_value}, PC2: {pc2_value}, Cluster: {cluster_label}")

# # Explore cluster characteristics (centroid values)
# centroids = kclu.cluster_centers_

# # Display top features contributing to each cluster
# print("Top features contributing to each cluster:")
# for i in range(num_clusters):
#     print(f"Cluster {i + 1}:")

#     # Get the indices of the top features for the current cluster
#     top_features_indices = np.argsort(centroids[i])[::-1][:5]

#     # Get the names of the top features
#     top_features = X.columns[top_features_indices].tolist()

#     print(top_features)

# # Set up the figure with subplots
# fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# # Plot the silhouette and Fowlkes-Mallows scores
# xs = range(2, 50)
# sils = []
# fms = []

# for i in xs:
#     kclu = KMeans(n_clusters=i, n_init=50)
#     kclu.fit(Xnew)
#     sils.append(metrics.silhouette_score(Xnew, kclu.labels_, metric='euclidean'))
#     fms.append(metrics.fowlkes_mallows_score(dfall_kmeans['oscar winners'], kclu.labels_))

# axs[0, 0].plot(xs, sils)
# axs[0, 0].plot(xs, fms)
# axs[0, 0].set_xlabel('Number of clusters (k)')
# axs[0, 0].set_ylabel('Score')
# axs[0, 0].set_title('Silhouette and Fowlkes-Mallows Scores')
# axs[0, 0].legend(['Silhouette', 'Fowlkes-Mallows'])

# # Plot the elbow curve
# inertias = []
# for i in range(1, 11):
#     kmeans = KMeans(n_clusters=i, n_init=50, random_state=42)
#     kmeans.fit(Xnew)
#     inertias.append(kmeans.inertia_)

# axs[0, 1].plot(range(1, 11), inertias, marker='o')
# axs[0, 1].set_xlabel('Number of Clusters')
# axs[0, 1].set_ylabel('Inertia')
# axs[0, 1].set_title('Elbow Method')

# # Plot silhouette scores
# silhouette_scores = []
# for i in range(2, 11):
#     kmeans = KMeans(n_clusters=i, n_init=50,  random_state=42)
#     kmeans.fit(Xnew)
#     silhouette_scores.append(metrics.silhouette_score(Xnew, kmeans.labels_))

# axs[1, 0].plot(range(2, 11), silhouette_scores, marker='o')
# axs[1, 0].set_xlabel('Number of Clusters')
# axs[1, 0].set_ylabel('Silhouette Score')
# axs[1, 0].set_title('Silhouette Score Method')

# # Plot the K-Means Clusters in PC1-PC2 Space
# # num_clusters = 5  # Replace with your actual number of clusters
# for i in range(num_clusters):
#     cluster_df = dfall_kmeans[dfall_kmeans['kmeans_cluster'] == i]
#     axs[1, 1].scatter(cluster_df['PC1'], cluster_df['PC2'], label=f'Cluster {i + 1}')

# axs[1, 1].set_xlabel("PC1")
# axs[1, 1].set_ylabel("PC2")
# axs[1, 1].set_title("K-Means Clusters in PC1-PC2 Space")
# axs[1, 1].legend()

# # Adjust layout to prevent clipping of titles and labels
# plt.tight_layout()

# # Show the plot
# plt.show()

# ----------------------------------------------------------------------------------TEST 1
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

iris = pd.read_excel(TRAIN_PATH_PROCESSED, sheet_name='Sheet1')
y, X = separateData(iris)

# scaler = StandardScaler()
scaler=MinMaxScaler()
# scaler=RobustScaler()
X_scaled = scaler.fit_transform(X)

# Example usage:
# iris = pd.read_excel(FULL_PREDICTIONS, sheet_name='Sheet1')
prediction_one_ids = get_rows_with_prediction_one(iris)

print("IDs of rows with prediction equal to 1:")
print(prediction_one_ids)

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
for index, row in dfall_kmeans.iterrows():
    point_id = index  # Assuming the index is the ID, you can adjust this based on your dataset
    pc1_value = row['PC1']
    pc2_value = row['PC2']
    cluster_label = row['kmeans_cluster'] + 1
    if point_id in prediction_one_ids:
        print(f"Point ID: {point_id+1}, PC1: {pc1_value}, PC2: {pc2_value}, Cluster: {cluster_label}")

# Explore cluster characteristics (centroid values)
centroids = kclu.cluster_centers_

# Display top features contributing to each cluster
print("Top features contributing to each cluster:")
for i in range(num_clusters):
    print(f"Cluster {i + 1}:")

    # Get the indices of the top features for the current cluster
    top_features_indices = np.argsort(centroids[i])[::-1][:5]

    # Get the names of the top features
    top_features = X.columns[top_features_indices].tolist()

    print(top_features)

# Set up the figure with subplots
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# Plot the silhouette and Fowlkes-Mallows scores
xs = range(2, 50)
sils = []
fms = []

for i in xs:
    kclu = KMeans(n_clusters=i, n_init='auto')
    kclu.fit(Xnew)
    sils.append(metrics.silhouette_score(Xnew, kclu.labels_, metric='euclidean'))
    fms.append(metrics.fowlkes_mallows_score(dfall_kmeans['oscar winners'], kclu.labels_))

axs[0, 0].plot(xs, sils)
axs[0, 0].plot(xs, fms)
axs[0, 0].set_xlabel('Number of clusters (k)')
axs[0, 0].set_ylabel('Score')
axs[0, 0].set_title('Silhouette and Fowlkes-Mallows Scores')
axs[0, 0].legend(['Silhouette', 'Fowlkes-Mallows'])

# Plot the elbow curve
inertias = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, n_init='auto', random_state=42)
    kmeans.fit(Xnew)
    inertias.append(kmeans.inertia_)

axs[0, 1].plot(range(1, 11), inertias, marker='o')
axs[0, 1].set_xlabel('Number of Clusters')
axs[0, 1].set_ylabel('Inertia')
axs[0, 1].set_title('Elbow Method')

# Plot silhouette scores
silhouette_scores = []
for i in range(2, 11):
    kmeans = KMeans(n_clusters=i, n_init='auto',  random_state=42)
    kmeans.fit(Xnew)
    silhouette_scores.append(metrics.silhouette_score(Xnew, kmeans.labels_))

axs[1, 0].plot(range(2, 11), silhouette_scores, marker='o')
axs[1, 0].set_xlabel('Number of Clusters')
axs[1, 0].set_ylabel('Silhouette Score')
axs[1, 0].set_title('Silhouette Score Method')

# Plot the K-Means Clusters in PC1-PC2 Space
for i in range(num_clusters):
    cluster_df = dfall_kmeans[dfall_kmeans['kmeans_cluster'] == i]
    axs[1, 1].scatter(cluster_df['PC1'], cluster_df['PC2'], label=f'Cluster {i + 1}')

axs[1, 1].set_xlabel("PC1")
axs[1, 1].set_ylabel("PC2")
axs[1, 1].set_title("K-Means Clusters in PC1-PC2 Space")
axs[1, 1].legend()

# Adjust layout to prevent clipping of titles and labels
plt.tight_layout()

# Show the plot
plt.show()

# ----------------------------------------------------------------------------------TEST 1
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn.metrics import confusion_matrix

TRAIN_PATH_PROCESSED = "./Data/moviesUpdated_processed.xlsx"
FULL_PREDICTIONS = './Data/full_predictions.xlsx'

def separateData(ds):
    if 'predictions' not in ds.columns:
        raise ValueError('predictions not in the dataset')
    target = ds['predictions']
    data = ds.drop(columns='predictions')
    # data = data.sort_index(axis=1)
    return target, data

def get_rows_with_prediction_one(df):
    if 'predictions' not in df.columns:
        raise ValueError('predictions not in the dataframe')

    prediction_one_ids = (df[df['predictions'] == 1].index).tolist()
    return prediction_one_ids

iris = pd.read_excel(FULL_PREDICTIONS, sheet_name='Sheet1')
y, X = separateData(iris)


# Example usage:
iris = pd.read_excel(FULL_PREDICTIONS, sheet_name='Sheet1')
prediction_one_ids = get_rows_with_prediction_one(iris)

print("IDs of rows with prediction equal to 1:")
print(prediction_one_ids)

# Perform PCA
pca = PCA(n_components=2)
Xnew = pca.fit_transform(X)
feature_names_pc1 = X.columns[np.argsort(pca.components_[0])[::-1]].tolist()

# Perform K-Means clustering
num_clusters = 4  # You can choose the number of clusters
kclu = KMeans(n_clusters=num_clusters, n_init='auto')
kclu.fit(Xnew)

# Add cluster labels to your dataframe
dfcluster_kmeans = pd.DataFrame(kclu.labels_, columns=['kmeans_cluster'])
dfpca = pd.DataFrame(Xnew, columns=["PC1", "PC2"])
dfclass = pd.DataFrame(y, columns=['predictions'])
dfall_kmeans = pd.concat([dfpca, dfclass, dfcluster_kmeans], axis=1)

# Print clustering stats
print("Confusion Matrix:")
print(confusion_matrix(dfall_kmeans['predictions'], dfall_kmeans['kmeans_cluster']))
print("Calinski-Harabasz Score:", metrics.calinski_harabasz_score(Xnew, dfall_kmeans['kmeans_cluster']))
print("Silhouette Score:", metrics.silhouette_score(Xnew, dfall_kmeans['kmeans_cluster'], metric='euclidean'))


# Explore cluster characteristics (centroid values)
centroids = kclu.cluster_centers_
for i in range(num_clusters):
    print(f"Cluster {i + 1} Centroid Values:")
    print("PC1:", centroids[i, 0])
    print("PC2:", centroids[i, 1])
    print()

# Visualize clusters in the reduced-dimensional space
for i in range(num_clusters):
    cluster_df = dfall_kmeans[dfall_kmeans['kmeans_cluster'] == i]
    plt.scatter(cluster_df['PC1'], cluster_df['PC2'], label=f'Cluster {i + 1}')

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("K-Means Clusters in PC1-PC2 Space")
plt.legend()
plt.show()

feature_importance_pc1 = pca.components_[0]
feature_importance_pc2 = pca.components_[1]

print("Top features contributing to PC1:")
top_features_pc1 = X.columns[np.argsort(feature_importance_pc1)[::-1]][:5].tolist()
print(top_features_pc1)

print("\nTop features contributing to PC2:")
top_features_pc2 = X.columns[np.argsort(feature_importance_pc2)[::-1]][:5].tolist()
print(top_features_pc2)
# print(0['kmeans_cluster'])
for index, row in dfall_kmeans.iterrows():
    # print(row)
    point_id = index  # Assuming the index is the ID, you can adjust this based on your dataset
    pc1_value = row['PC1']
    pc2_value = row['PC2']
    cluster_label = row['kmeans_cluster']+1
    if point_id in prediction_one_ids:
      print(f"Point ID: {point_id}, PC1: {pc1_value}, PC2: {pc2_value}, Cluster: {cluster_label}")

# Explore cluster characteristics (centroid values)
centroids = kclu.cluster_centers_

# Display top features contributing to each cluster
print("Top features contributing to each cluster:")
for i in range(num_clusters):
    print(f"Cluster {i + 1}:")

    # Get the indices of the top features for the current cluster
    top_features_indices = np.argsort(centroids[i])[::-3][:4]

    # Get the names of the top features
    top_features = X.columns[top_features_indices].tolist()

    print(top_features)



# -------------------------------------------------------------------------BEGGINING
# import pandas as pd
# import matplotlib.pyplot as plt
# import sklearn
# from sklearn import datasets
# from sklearn.cluster import KMeans
# from sklearn.decomposition import PCA
# from sklearn import metrics
# from sklearn.metrics import pairwise_distances
# from sklearn.metrics import confusion_matrix

# TRAIN_PATH_PROCESSED="./Data/moviesUpdated_processed.xlsx"
# FULL_PREDICTIONS='./Data/full_predictions.xlsx'

# def seperateData(ds):
#   if 'predictions' not in ds.columns:
#     raise ValueError('predictions not in the dataset')
#   target=ds['predictions']
#   data=ds.drop(columns='predictions')
#   data=data.sort_index(axis=1)
#   return (target,data)

# iris = pd.read_excel(FULL_PREDICTIONS, sheet_name = 'Sheet1')
# y,X=seperateData(iris)
# # X = iris.data
# # y = iris.target

# pca = PCA(n_components=2)
# Xnew = pca.fit(X).transform(X)

# dfpca = pd.DataFrame(Xnew, columns=["PC1", "PC2"])
# dfclass = pd.DataFrame(y, columns=['predictions'])
# dfpca = pd.concat([dfpca, dfclass], axis=1)

# num_clusters=4 # 4 is the best so far
# kclu = KMeans(n_clusters=num_clusters,n_init='auto')
# kclu.fit(Xnew)

# dfcluster = pd.DataFrame(kclu.labels_, columns=['cluster'])
# dfall = pd.concat([dfpca, dfcluster], axis=1)

# print(confusion_matrix(dfall['predictions'], dfall["cluster"]))
# print(metrics.calinski_harabasz_score(Xnew, dfall['cluster']) )
# print(metrics.silhouette_score(Xnew, dfall['cluster'], metric='euclidean'))


# # Plot the k-means clusters
# for i in range(num_clusters):
#     cluster_df = dfall[dfall['cluster'] == i]
#     plt.plot(cluster_df['PC1'], cluster_df['PC2'], 'o', label=f'Cluster {i + 1}')

# plt.xlabel(dfpca.columns[0])
# plt.ylabel(dfpca.columns[1])
# plt.title(f'{num_clusters} Clusters k-means')
# plt.legend()
# plt.show()

# # xs=range(2,50)
# # sils=[]
# # fms=[]

# # for i in xs:
# #     kclu=KMeans(n_clusters=i)
# #     kclu.fit(X)
# #     sils.append(metrics.silhouette_score(X, kclu.labels_, metric='euclidean'))
# #     fms.append(metrics.fowlkes_mallows_score(dfall['predictions'], kclu.labels_))
# # plt.plot(xs,sils)
# # plt.plot(xs,fms)
# # plt.xlabel('Number of clusters (k)')
# # plt.ylabel('Silhouette')
# # plt.title('Evaluate how k affects cluster validity')
# # plt.legend(['silhoute','folkes mallows'])
# # plt.show()
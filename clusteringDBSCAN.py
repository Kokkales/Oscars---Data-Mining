# ----------------------------------------------------------------------------------TEST 1
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn.metrics import confusion_matrix

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

# Example usage:
# iris = pd.read_excel(FULL_PREDICTIONS, sheet_name='Sheet1')
prediction_one_ids = get_rows_with_prediction_one(iris)

print("IDs of rows with prediction equal to 1:")
print(prediction_one_ids)

# Perform PCA
pca = PCA(n_components=2)
Xnew = pca.fit_transform(X)
feature_names_pc1 = X.columns[np.argsort(pca.components_[0])[::-1]].tolist()

# Perform DBSCAN clustering
eps_value = float(sys.argv[1])  # You can choose the epsilon value
min_samples_value = int(sys.argv[2])  # You can choose the min_samples value
dbscan = DBSCAN(eps=eps_value, min_samples=min_samples_value)
dbscan_labels = dbscan.fit_predict(Xnew)

# Check if there is only one cluster (which results in a ValueError for silhouette score)
unique_labels = np.unique(dbscan_labels)
if len(unique_labels) == 1:
    print("DBSCAN resulted in only one cluster. Silhouette score is not applicable.")
else:
    # Add cluster labels to your dataframe
    dfcluster_dbscan = pd.DataFrame(dbscan_labels, columns=['dbscan_cluster'])
    dfpca = pd.DataFrame(Xnew, columns=["PC1", "PC2"])
    dfclass = pd.DataFrame(y, columns=['oscar winners'])
    dfall_dbscan = pd.concat([dfpca, dfclass, dfcluster_dbscan], axis=1)

    # Print clustering stats
    print("Confusion Matrix:")
    print(confusion_matrix(dfall_dbscan['oscar winners'], dfall_dbscan['dbscan_cluster']))

    # Silhouette score is not applicable if there is only one cluster
    if len(unique_labels) > 1:
        print("Silhouette Score:", metrics.silhouette_score(Xnew, dfall_dbscan['dbscan_cluster'], metric='euclidean'))

    # Explore cluster characteristics (no explicit centroids in DBSCAN)
    # Display top features contributing to each cluster
    print("Top features contributing to each cluster:")
    unique_clusters_dbscan = np.unique(dbscan_labels)
    for i, cluster_id in enumerate(unique_clusters_dbscan):
        if cluster_id == -1:  # -1 represents noise points in DBSCAN
            continue

        cluster_df = dfall_dbscan[dfall_dbscan['dbscan_cluster'] == cluster_id]
        print(f"Cluster {i + 1}:")

        # Get the indices of the top features for the current cluster
        top_features_indices = np.argsort(np.mean(cluster_df[['PC1', 'PC2']].values, axis=0))[::-1][:5]

        # Get the names of the top features
        top_features = X.columns[top_features_indices].tolist()
        print(top_features)

    # Set up the figure with subplots
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    # Plot the DBSCAN Clusters in PC1-PC2 Space
    for i, cluster_id in enumerate(unique_clusters_dbscan):
        cluster_df = dfall_dbscan[dfall_dbscan['dbscan_cluster'] == cluster_id]
        if cluster_id == -1:
            axs[1, 1].scatter(cluster_df['PC1'], cluster_df['PC2'], label=f'Noise Points')
        else:
            axs[1, 1].scatter(cluster_df['PC1'], cluster_df['PC2'], label=f'Cluster {i + 1}')

    axs[1, 1].set_xlabel("PC1")
    axs[1, 1].set_ylabel("PC2")
    axs[1, 1].set_title("DBSCAN Clusters in PC1-PC2 Space")
    axs[1, 1].legend()

    # Adjust layout to prevent clipping of titles and labels
    plt.tight_layout()

    # Show the plot
    plt.show()

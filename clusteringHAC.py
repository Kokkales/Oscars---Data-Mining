import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler

TRAIN_PATH_PROCESSED = "./Data/moviesUpdated_processed.xlsx"
FULL_PREDICTIONS = './Data/full_predictions.xlsx'

def separateData(dataset):
    if 'oscar winners' not in dataset.columns:
        raise ValueError('oscar winners not in the dataset')
    target = dataset['oscar winners']
    data = dataset.drop(columns='oscar winners')
    return target, data

def getRowsWithPredictionOne(dataframe):
    if 'oscar winners' not in dataframe.columns:
        raise ValueError('oscar winners not in the dataframe')

    predictionOneIds = (dataframe[dataframe['oscar winners'] == 1].index).tolist()
    return predictionOneIds

iris = pd.read_excel(TRAIN_PATH_PROCESSED, sheet_name='Sheet1')
y, X = separateData(iris)

scaler = RobustScaler()
if sys.argv[2] == 'ss':
    scaler = StandardScaler()
elif sys.argv[2] == 'rs':
    scaler = RobustScaler()
else:
    scaler = MinMaxScaler()
Xscaled = scaler.fit_transform(X)

predictionOneIds = getRowsWithPredictionOne(iris)

print("IDs of rows with prediction equal to 1:")
print(predictionOneIds)

# Perform PCA
pca = PCA(n_components=2)
Xnew = pca.fit_transform(Xscaled)
featureNamesPC1 = X.columns[np.argsort(pca.components_[0])[::-1]].tolist()

# Perform Hierarchical Agglomerative Clustering
numClusters = int(sys.argv[1])  # You can choose the number of clusters
hac = AgglomerativeClustering(n_clusters=numClusters)
hacLabels = hac.fit_predict(Xnew)

# Add cluster labels to your dataframe
dfClusterHAC = pd.DataFrame(hacLabels, columns=['hacCluster'])
dfPCA = pd.DataFrame(Xnew, columns=["PC1", "PC2"])
dfClass = pd.DataFrame(y, columns=['oscar winners'])
dfAllHAC = pd.concat([dfPCA, dfClass, dfClusterHAC], axis=1)

# Print clustering stats
print("Confusion Matrix:")
print(confusion_matrix(dfAllHAC['oscar winners'], dfAllHAC['hacCluster']))
print("Calinski-Harabasz Score:", metrics.calinski_harabasz_score(Xnew, dfAllHAC['hacCluster']))
print("Silhouette Score:", metrics.silhouette_score(Xnew, dfAllHAC['hacCluster'], metric='euclidean'))

# Explore cluster characteristics (centroid values)
# HAC doesn't have explicit centroids, so we'll use the mean values of each cluster
centroids = dfAllHAC.groupby('hacCluster').mean().values

for i in range(numClusters):
    print(f"Cluster {i + 1} Centroid Values:")
    print("PC1:", centroids[i, 0])
    print("PC2:", centroids[i, 1])
    print()

# Display top features contributing to each cluster
print("Top features contributing to each cluster:")
for i in range(numClusters):
    print(f"Cluster {i + 1}:")

    # Get the indices of the top features for the current cluster
    topFeaturesIndices = np.argsort(centroids[i])[::-1][:5]

    # Get the names of the top features
    topFeatures = X.columns[topFeaturesIndices].tolist()

    print(topFeatures)

# Loop through the HAC clusters and print information for points with prediction equal to 1
pc1Osc = []
pc2Osc = []
for index, row in dfAllHAC.iterrows():
    pointId = index  # Assuming the index is the ID, you can adjust this based on your dataset
    pc1Value = row['PC1']
    pc2Value = row['PC2']
    clusterLabel = row['hacCluster'] + 1

    if pointId in predictionOneIds:
        pc1Osc.append(pc1Value)
        pc2Osc.append(pc2Value)
        # print(f"Point ID: {pointId+1}, PC1: {pc1Value}, PC2: {pc2Value}, Cluster: {clusterLabel}")

# Set up the figure with subplots
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# Plot the silhouette and Fowlkes-Mallows scores
xs = range(2, 50)
sils = []
fms = []

for i in xs:
    hac = AgglomerativeClustering(n_clusters=i)
    hacLabels = hac.fit_predict(Xnew)
    sils.append(metrics.silhouette_score(Xnew, hacLabels, metric='euclidean'))
    fms.append(metrics.fowlkes_mallows_score(dfAllHAC['oscar winners'], hacLabels))

axs[0].plot(xs, sils)
axs[0].plot(xs, fms)
axs[0].set_xlabel('Number of clusters (k)')
axs[0].set_ylabel('Score')
axs[0].set_title('Silhouette and Fowlkes-Mallows Scores')
axs[0].legend(['Silhouette', 'Fowlkes-Mallows'])
# Plot silhouette scores
silhouetteScores = []
for i in range(2, 11):
    hac = AgglomerativeClustering(n_clusters=i)
    hacLabels = hac.fit_predict(Xnew)
    silhouetteScores.append(metrics.silhouette_score(Xnew, hacLabels, metric='euclidean'))

axs[1].plot(range(2, 11), silhouetteScores, marker='o')
axs[1].set_xlabel('Number of Clusters')
axs[1].set_ylabel('Silhouette Score')
axs[1].set_title('Silhouette Score Method')

# Plot the HAC Clusters in PC1-PC2 Space
# numClusters = 5  # Replace with your actual number of clusters
for i in range(numClusters):
    clusterDF = dfAllHAC[dfAllHAC['hacCluster'] == i]
    axs[2].scatter(clusterDF['PC1'], clusterDF['PC2'], label=f'Cluster {i + 1}')
axs[2].scatter(pc1Osc, pc2Osc, label=f'Oscars', color='red')
axs[2].set_xlabel("PC1")
axs[2].set_ylabel("PC2")
axs[2].set_title("HAC Clusters in PC1-PC2 Space")
axs[2].legend()

# Adjust layout to prevent clipping of titles and labels
plt.tight_layout()

# Show the plot
plt.show()

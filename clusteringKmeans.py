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

if len(sys.argv) != 3:
    print("Usage: python3 clustering.py <cluster algorithm> <cluster number> <ss/rs/mm>")
    sys.exit(1)

iris = pd.read_excel(TRAIN_PATH_PROCESSED, sheet_name='Sheet1')
y, X = separateData(iris)
predictionOneIds = getRowsWithPredictionOne(iris)

if sys.argv[2] == 'ss':
    scaler = StandardScaler()
elif sys.argv[2] == 'rs':
    scaler = RobustScaler()
else:
    scaler = MinMaxScaler()
Xscaled = scaler.fit_transform(X)

# Perform PCA on the scaled data
pca = PCA(n_components=2)
Xnew = pca.fit_transform(Xscaled)
featureNamesPC1 = X.columns[np.argsort(pca.components_[0])[::-1]].tolist()

# Perform K-Means clustering on the scaled data
numClusters = int(sys.argv[1])  # You can choose the number of clusters
kMeans = KMeans(n_clusters=numClusters, n_init='auto', random_state=42)
kMeans.fit(Xnew)

# Add cluster labels to your dataframe
dfClusterKMeans = pd.DataFrame(kMeans.labels_, columns=['kMeansCluster'])
dfPCA = pd.DataFrame(Xnew, columns=["PC1", "PC2"])
dfClass = pd.DataFrame(y, columns=['oscar winners'])
dfAllKMeans = pd.concat([dfPCA, dfClass, dfClusterKMeans], axis=1)

# Print clustering stats
print("Confusion Matrix:")
print(confusion_matrix(dfAllKMeans['oscar winners'], dfAllKMeans['kMeansCluster']))
print("Calinski-Harabasz Score:", metrics.calinski_harabasz_score(Xnew, dfAllKMeans['kMeansCluster']))
print("Silhouette Score:", metrics.silhouette_score(Xnew, dfAllKMeans['kMeansCluster'], metric='euclidean'))

# Explore cluster characteristics (centroid values)
centroids = kMeans.cluster_centers_
for i in range(numClusters):
    print(f"Cluster {i + 1} Centroid Values:")
    print("PC1:", centroids[i, 0])
    print("PC2:", centroids[i, 1])
    print()

featureImportancePC1 = pca.components_[0]
featureImportancePC2 = pca.components_[1]

print("Top features contributing to PC1:")
topFeaturesPC1 = X.columns[np.argsort(featureImportancePC1)[::-1]][:5].tolist()
print(topFeaturesPC1)

print("\nTop features contributing to PC2:")
topFeaturesPC2 = X.columns[np.argsort(featureImportancePC2)[::-1]][:5].tolist()
print(topFeaturesPC2)

# Loop through the K-Means clusters and print information for points with prediction equal to 1
pc1Osc = []
pc2Osc = []
for index, row in dfAllKMeans.iterrows():
    pointId = index
    pc1Value = row['PC1']
    pc2Value = row['PC2']
    clusterLabel = row['kMeansCluster'] + 1
    if pointId in predictionOneIds:
        pc1Osc.append(pc1Value)
        pc2Osc.append(pc2Value)

centroids = kMeans.cluster_centers_
# Display top features contributing to each cluster
print("Top features contributing to each cluster:")
for i in range(numClusters):
    print(f"Cluster {i + 1}:")
    topFeaturesIndices = np.argsort(centroids[i])[::-1][:7]
    topFeatures = X.columns[topFeaturesIndices].tolist()
    print(topFeatures)

fig, axs = plt.subplots(1, 3, figsize=(15, 5))
# Plot the silhouette score comparing to Fowlkes-Mallows score
xs = range(2, 50)
sils = []
fms = []

for i in xs:
    kMeans = KMeans(n_clusters=i, n_init='auto')
    kMeans.fit(Xnew)
    sils.append(metrics.silhouette_score(Xnew, kMeans.labels_, metric='euclidean'))
    fms.append(metrics.fowlkes_mallows_score(dfAllKMeans['oscar winners'], kMeans.labels_))
axs[0].plot(xs, sils)
axs[0].plot(xs, fms)
axs[0].set_xlabel('Number of clusters (k)')
axs[0].set_ylabel('Score')
axs[0].set_title('Silhouette and Fowlkes-Mallows Scores')
axs[0].legend(['Silhouette', 'Fowlkes-Mallows'])

# Plot silhouette scores
silhouetteScores = []
for i in range(2, 11):
    kMeans = KMeans(n_clusters=i, n_init='auto', random_state=42)
    kMeans.fit(Xnew)
    silhouetteScores.append(metrics.silhouette_score(Xnew, kMeans.labels_))
axs[1].plot(range(2, 11), silhouetteScores, marker='o')
axs[1].set_xlabel('Number of Clusters')
axs[1].set_ylabel('Silhouette Score')
axs[1].set_title('Silhouette Score')

# Plot clusters according to principal components 1 and 2
for i in range(numClusters):
    clusterDF = dfAllKMeans[dfAllKMeans['kMeansCluster'] == i]
    axs[2].scatter(clusterDF['PC1'], clusterDF['PC2'], label=f'Cluster {i + 1}')
axs[2].scatter(pc1Osc, pc2Osc, label=f'Oscars', color='red')
axs[2].set_xlabel("PC1")
axs[2].set_ylabel("PC2")
axs[2].set_title("K-Means Clusters in PC1-PC2 Space")
axs[2].legend()

plt.tight_layout()
plt.show()
plt.close()

import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import Birch
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import silhouette_score

TRAIN_PATH_PROCESSED = "./Data/moviesUpdated_processed.xlsx"
FULL_PREDICTIONS = './Data/full_predictions.xlsx'


def separateData(dataFrame):
    if 'oscar winners' not in dataFrame.columns:
        raise ValueError('oscar winners not in the dataset')
    target = dataFrame['oscar winners']
    data = dataFrame.drop(columns='oscar winners')
    return target, data

def scale(X):
    if sys.argv[3] == 'ss':
        scaler = StandardScaler()
    elif sys.argv[3] == 'rs':
        scaler = RobustScaler()
    else:
        scaler = MinMaxScaler()
    Xscaled = scaler.fit_transform(X)
    return Xscaled

def loadDataset():
    dataset = pd.read_excel(TRAIN_PATH_PROCESSED, sheet_name='Sheet1')
    y, X = separateData(dataset)
    return y,X,dataset

def getRowsWithPredictionOne(dataFrame):
    if 'oscar winners' not in dataFrame.columns:
        raise ValueError('oscar winners not in the dataframe')

    predictionOneIds = (dataFrame[dataFrame['oscar winners'] == 1].index).tolist()
    return predictionOneIds

class Clustering:

    def __init__(self,y,X,Xscaled,dataFrame):
        self.y=y
        self.X=X
        self.Xscaled=Xscaled
        self.dataFrame=dataFrame

    def updateDataframe():
        print('dataframe Updated')

    def plotDiagrams(self,dfAllClusters,Xnew,pc1Osc,pc2Osc,numClusters):
        fig, axs = plt.subplots(1, 2, figsize=(15, 5))

        # Plot the silhouette score comparing to Fowlkes-Mallows score
        xs = range(2, 50)
        sils = []
        fms = []

        for i in xs:
            clusteringAlgorithm = AgglomerativeClustering(n_clusters=i)
            clusterLabels = clusteringAlgorithm.fit_predict(Xnew)
            sils.append(metrics.silhouette_score(Xnew, clusterLabels, metric='euclidean'))
            fms.append(metrics.fowlkes_mallows_score(dfAllClusters['oscar winners'], clusterLabels))

        axs[0].plot(xs, sils)
        axs[0].plot(xs, fms)
        axs[0].set_xlabel('Number of clusters (k)')
        axs[0].set_ylabel('Score')
        axs[0].set_title('Silhouette and Fowlkes-Mallows Scores')
        axs[0].legend(['Silhouette', 'Fowlkes-Mallows'])

        # Plot the Clusters PC1=x PC2=y
        for i in range(numClusters):
            clusterDF = dfAllClusters[dfAllClusters['cluster'] == i]
            axs[1].scatter(clusterDF['PC1'], clusterDF['PC2'], label=f'Cluster {i + 1}')
        axs[1].scatter(pc1Osc, pc2Osc, label=f'Oscars', color='red')
        axs[1].set_xlabel("PC1")
        axs[1].set_ylabel("PC2")
        axs[1].set_title("Clusters in PC1-PC2 Space")
        axs[1].legend()

        plt.tight_layout()
        plt.show()



    def printStats(self,y,dfAllClusters,Xnew,X,clusteringAlgorithm,numClusters,pca,alg):
        # Print clustering stats
        print("Confusion Matrix:")
        print(confusion_matrix(dfAllClusters['oscar winners'], dfAllClusters['cluster']))
        print("Calinski-Harabasz Score:", metrics.calinski_harabasz_score(Xnew, dfAllClusters['cluster']))
        print("Silhouette Score:", metrics.silhouette_score(Xnew, dfAllClusters['cluster'], metric='euclidean'))

        # Explore cluster characteristics (centroid values)
        if alg!='KM':
            centroids = dfAllClusters.groupby('cluster').mean().values
        else:
            centroids = clusteringAlgorithm.cluster_centers_

        # for i in range(numClusters):
        #     print(f"Cluster {i + 1} Centroid Values:")
        #     print("PC1:", centroids[i, 0])
        #     print("PC2:", centroids[i, 1])
        #     print()

        featureImportancePC1 = pca.components_[0]
        featureImportancePC2 = pca.components_[1]

        print("Top features contributing to PC1:")
        topFeaturesPC1 = X.columns[np.argsort(featureImportancePC1)[::-1]][:5].tolist()
        print(topFeaturesPC1)

        print("\nTop features contributing to PC2:")
        topFeaturesPC2 = X.columns[np.argsort(featureImportancePC2)[::-1]][:5].tolist()
        print(topFeaturesPC2)

        # Loop through the clusters and print information for points with prediction equal to 1
        predictionOneIds = getRowsWithPredictionOne(y)
        pc1Osc = []
        pc2Osc = []
        for index, row in dfAllClusters.iterrows():
            pointId = index
            pc1Value = row['PC1']
            pc2Value = row['PC2']
            clusterLabel = row['cluster'] + 1
            if pointId in predictionOneIds:
                pc1Osc.append(pc1Value)
                pc2Osc.append(pc2Value)

        # centroids = clusteringAlgorithm.cluster_centers_
        # Display top features contributing to each cluster
        print("Top features contributing to each cluster:")
        for i in range(numClusters):
            print(f"Cluster {i + 1}:")
            topFeaturesIndices = np.argsort(centroids[i])[::-1][:7]
            topFeatures = X.columns[topFeaturesIndices].tolist()
            print(topFeatures)

        return pc1Osc,pc2Osc



    def executeClustering(self,alg='KM',numClusters=2):
        # Perform PCA on the scaled data
        pca = PCA(n_components=2)
        Xnew = pca.fit_transform(self.Xscaled)
        # print("PCA Components:")
        # print(pca.components_)
        y=pd.DataFrame(self.y)
        if alg=='KM':
            clusteringAlgorithm = KMeans(n_clusters=numClusters, n_init='auto', random_state=42)
            print('KMeans')
        elif alg=='HAC':
            clusteringAlgorithm = AgglomerativeClustering(n_clusters=numClusters)
            print('HAC')
        elif alg=='DBSCAN':
            eps_value = 0.5
            clusteringAlgorithm = DBSCAN(eps=eps_value, min_samples=numClusters)
            print('DBSCAN')
        else:
            clusteringAlgorithm = Birch(n_clusters=numClusters)
            print('Birch')
        clusterLabels = clusteringAlgorithm.fit_predict(Xnew)
        # Add cluster labels in the dataframe
        dfCluster = pd.DataFrame(clusterLabels, columns=['cluster'])
        dfPCA = pd.DataFrame(Xnew, columns=["PC1", "PC2"])
        dfClass = pd.DataFrame(self.y, columns=['oscar winners'])
        dfAllClusters = pd.concat([dfPCA, dfClass, dfCluster], axis=1)
        pc1Osc,pc2Osc=self.printStats(y,dfAllClusters,Xnew,self.X,clusteringAlgorithm,numClusters,pca,alg)
        self.plotDiagrams(dfAllClusters,Xnew,pc1Osc,pc2Osc,numClusters)

if __name__=='__main__':
    if len(sys.argv) != 4:
        print("Usage: python3 clustering.py <clusterAlgorithm> <clustersNumber> <ss/rs/mm>")
        sys.exit(1)
    y,X,dataSet=loadDataset()
    Xscaled=scale(X)


    cl=Clustering(y=y,X=X,Xscaled=Xscaled,dataFrame=dataSet)
    cl.executeClustering(alg=sys.argv[1],numClusters=int(sys.argv[2]))
    print('CLUSTERING IS EXECUTING')

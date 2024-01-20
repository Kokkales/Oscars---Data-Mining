from clustering import Clustering
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler
import sys

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


# STEPS
if len(sys.argv) != 4:
    print("Usage: python3 clustering.py <clusterAlgorithm> <clustersNumber> <ss/rs/mm>")
    sys.exit(1)
y,X,dataSet=loadDataset()
X=scale(X)

cl=Clustering(y=y,X=X,dataFrame=dataSet)
cl.executeClustering(alg=sys.argv[1],numClusters=int(sys.argv[2]))
print('FINISHED')
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn.metrics import pairwise_distances
from sklearn.metrics import confusion_matrix

TRAIN_PATH_PROCESSED="./Data/moviesUpdated_processed.xlsx"
FULL_PREDICTIONS='./Data/full_predictions.xlsx'

def seperateData(ds):
  if 'predictions' not in ds.columns:
    raise ValueError('predictions not in the dataset')
  target=ds['predictions']
  data=ds.drop(columns='predictions')
  data=data.sort_index(axis=1)
  return (target,data)

iris = pd.read_excel(FULL_PREDICTIONS, sheet_name = 'Sheet1')
y,X=seperateData(iris)
# X = iris.data
# y = iris.target

pca = PCA(n_components=2)
Xnew = pca.fit(X).transform(X)

dfpca = pd.DataFrame(Xnew, columns=["PC1", "PC2"])
dfclass = pd.DataFrame(y, columns=['predictions'])
dfpca = pd.concat([dfpca, dfclass], axis=1)

num_clusters=4 # 4 is the best so far
kclu = KMeans(n_clusters=num_clusters,n_init='auto')
kclu.fit(Xnew)

dfcluster = pd.DataFrame(kclu.labels_, columns=['cluster'])
dfall = pd.concat([dfpca, dfcluster], axis=1)

print(confusion_matrix(dfall['predictions'], dfall["cluster"]))
print(metrics.calinski_harabasz_score(Xnew, dfall['cluster']) )
print(metrics.silhouette_score(Xnew, dfall['cluster'], metric='euclidean'))


# Plot the k-means clusters
for i in range(num_clusters):
    cluster_df = dfall[dfall['cluster'] == i]
    plt.plot(cluster_df['PC1'], cluster_df['PC2'], 'o', label=f'Cluster {i + 1}')

plt.xlabel(dfpca.columns[0])
plt.ylabel(dfpca.columns[1])
plt.title(f'{num_clusters} Clusters k-means')
plt.legend()
plt.show()

# xs=range(2,50)
# sils=[]
# fms=[]

# for i in xs:
#     kclu=KMeans(n_clusters=i)
#     kclu.fit(X)
#     sils.append(metrics.silhouette_score(X, kclu.labels_, metric='euclidean'))
#     fms.append(metrics.fowlkes_mallows_score(dfall['predictions'], kclu.labels_))
# plt.plot(xs,sils)
# plt.plot(xs,fms)
# plt.xlabel('Number of clusters (k)')
# plt.ylabel('Silhouette')
# plt.title('Evaluate how k affects cluster validity')
# plt.legend(['silhoute','folkes mallows'])
# plt.show()
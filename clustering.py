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

def seperateData(ds):
  if 'oscar winners' not in ds.columns:
    raise ValueError('Oscar winners not in the dataset')
  target=ds['oscar winners']
  data=ds.drop(columns='oscar winners')
  data=data.sort_index(axis=1)
  return (target,data)

iris = pd.read_excel(TRAIN_PATH_PROCESSED, sheet_name = 'Sheet1')
y,X=seperateData(iris)
# X = iris.data
# y = iris.target

pca = PCA(n_components=2)
Xnew = pca.fit(X).transform(X)

dfpca = pd.DataFrame(Xnew, columns=["PC1", "PC2"])
dfclass = pd.DataFrame(y, columns=['oscar winners'])
dfpca = pd.concat([dfpca, dfclass], axis=1)

kclu = KMeans(n_clusters=3,n_init='auto')
kclu.fit(Xnew)

dfcluster = pd.DataFrame(kclu.labels_, columns=['cluster'])
dfall = pd.concat([dfpca, dfcluster], axis=1)

print(confusion_matrix(dfall['oscar winners'], dfall["cluster"]))
print(metrics.calinski_harabasz_score(Xnew, dfall['cluster']) )
print(metrics.silhouette_score(Xnew, dfall['cluster'], metric='euclidean'))

# plot
print(dfpca.columns[2])
df0=dfall[dfall['oscar winners']==0]
df1=dfall[dfall['oscar winners']==1]
df2=dfall[dfall['oscar winners']==2]

plt.plot(df0['PC1'],df0['PC2'],'bv')
plt.plot(df1['PC1'],df1['PC2'],'ro')
plt.plot(df2['PC1'],df2['PC2'],'gd')

plt.xlabel(dfpca.columns[0])
plt.ylabel(dfpca.columns[1])
plt.title('3 Classes Iris')
plt.show()

df0=dfall[dfall['cluster']==0]
df1=dfall[dfall['cluster']==1]
df2=dfall[dfall['cluster']==2]

plt.plot(df0['PC1'],df0['PC2'],'bv')
plt.plot(df1['PC1'],df1['PC2'],'ro')
plt.plot(df2['PC1'],df2['PC2'],'gd')

plt.xlabel(dfpca.columns[0])
plt.ylabel(dfpca.columns[1])
plt.title('3 Clusters k-means')
plt.show()

xs=range(2,50)
sils=[]
fms=[]

for i in xs:
    kclu=KMeans(n_clusters=i)
    kclu.fit(X)
    sils.append(metrics.silhouette_score(X, kclu.labels_, metric='euclidean'))
    fms.append(metrics.fowlkes_mallows_score(dfall['oscar winners'], kclu.labels_))
plt.plot(xs,sils)
plt.plot(xs,fms)
plt.xlabel('Number of clusters (k)')
plt.ylabel('Silhouette')
plt.title('Evaluate how k affects cluster validity')
plt.legend(['silhoute','folkes mallows'])
plt.show()
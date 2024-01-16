import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn.metrics import pairwise_distances
from sklearn.metrics import confusion_matrix
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

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

Xnew = StandardScaler().fit_transform(Xnew)
xs=range(4,12)
clus=[]

for i in xs:
    kclu=DBSCAN(eps=0.5, min_samples=i)
    kclu.fit(X)
    n_clusters_ = len(set(kclu.labels_)) - (1 if -1 in kclu.labels_ else 0)
    clus.append(n_clusters_)

plt.plot(xs,clus)
plt.xlabel('MINPTS for eps=0.5')
plt.ylabel('Clusters')
plt.title('Evaluate how MINPTS affects number of clusters')
plt.legend(['clusters'])
plt.show()
kclu=DBSCAN(eps=0.5, min_samples=8)

kclu.fit(X)
dfcluster=pd.DataFrame(kclu.labels_,columns=['cluster'])
dfall=pd.concat([dfpca,dfcluster],axis=1)
dfall = dfall[dfall.cluster != -1]
print(confusion_matrix(dfall['oscar winners'], dfall['cluster']))


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
df0=dfall[dfall['cluster']==0]
df1=dfall[dfall['cluster']==1]
df2=dfall[dfall['cluster']==2]
plt.plot(df0['PC1'],df0['PC2'],'bv')
plt.plot(df1['PC1'],df1['PC2'],'ro')
plt.plot(df2['PC1'],df2['PC2'],'gd')


plt.xlabel(iris.feature_names[1])
plt.ylabel(iris.feature_names[2])
plt.title('DBSCAN Clusters')
plt.show()

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
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
    data = data.sort_index(axis=1)
    return target, data

iris = pd.read_excel(FULL_PREDICTIONS, sheet_name='Sheet1')
y, X = separateData(iris)

pca = PCA(n_components=2)
Xnew = pca.fit_transform(X)

dfpca = pd.DataFrame(Xnew, columns=["PC1", "PC2"])
dfclass = pd.DataFrame(y, columns=['predictions'])
dfpca = pd.concat([dfpca, dfclass], axis=1)

gmm_clu = GaussianMixture(n_components=4)  # Adjust parameters as needed
gmm_labels = gmm_clu.fit_predict(Xnew)

dfcluster = pd.DataFrame(gmm_labels, columns=['cluster'])
dfall = pd.concat([dfpca, dfcluster], axis=1)

print(confusion_matrix(dfall['predictions'], dfall["cluster"]))
print(metrics.calinski_harabasz_score(Xnew, dfall['cluster']))
print(metrics.silhouette_score(Xnew, dfall['cluster'], metric='euclidean'))

# Plot the GMM clusters
unique_clusters = set(gmm_labels)
for cluster_id in unique_clusters:
    cluster_df = dfall[dfall['cluster'] == cluster_id]
    plt.plot(cluster_df['PC1'], cluster_df['PC2'], 'o', label=f'Cluster {cluster_id}')

plt.xlabel(dfpca.columns[0])
plt.ylabel(dfpca.columns[1])
plt.title('Gaussian Mixture Model Clustering')
plt.legend()
plt.show()

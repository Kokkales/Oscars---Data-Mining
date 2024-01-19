import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AffinityPropagation
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

def get_rows_with_prediction_one(df):
    if 'predictions' not in df.columns:
        raise ValueError('predictions not in the dataframe')

    prediction_one_ids = (df[df['predictions'] == 1].index).tolist()
    return prediction_one_ids

iris = pd.read_excel(FULL_PREDICTIONS, sheet_name='Sheet1')
y, X = separateData(iris)

pca = PCA(n_components=2)
Xnew = pca.fit_transform(X)

dfpca = pd.DataFrame(Xnew, columns=["PC1", "PC2"])
dfclass = pd.DataFrame(y, columns=['predictions'])
dfpca = pd.concat([dfpca, dfclass], axis=1)

affinity_propagation_clu = AffinityPropagation(damping=0.7)  # Adjust parameters as needed
affinity_propagation_labels = affinity_propagation_clu.fit_predict(Xnew)

dfcluster = pd.DataFrame(affinity_propagation_labels, columns=['cluster'])
dfall = pd.concat([dfpca, dfcluster], axis=1)

# Print clustering stats
print(confusion_matrix(dfall['predictions'], dfall["cluster"]))
print(metrics.calinski_harabasz_score(Xnew, dfall['cluster']))
print(metrics.silhouette_score(Xnew, dfall['cluster'], metric='euclidean'))

# Print points with prediction equal to 1
prediction_one_ids = get_rows_with_prediction_one(iris)
for index, row in dfall.iterrows():
    point_id = index
    pc1_value = row['PC1']
    pc2_value = row['PC2']
    cluster_label = row['cluster']

    if point_id in prediction_one_ids:
        print(f"Point ID: {point_id+1}, PC1: {pc1_value}, PC2: {pc2_value}, Cluster: {cluster_label}")

# Plot the Affinity Propagation clusters
unique_clusters = set(affinity_propagation_labels)
for cluster_id in unique_clusters:
    cluster_df = dfall[dfall['cluster'] == cluster_id]
    plt.plot(cluster_df['PC1'], cluster_df['PC2'], 'o', label=f'Cluster {cluster_id}')

plt.xlabel(dfpca.columns[0])
plt.ylabel(dfpca.columns[1])
plt.title('Affinity Propagation Clustering')
plt.legend()
plt.show()

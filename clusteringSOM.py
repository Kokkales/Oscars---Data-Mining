import pandas as pd
import matplotlib.pyplot as plt
from minisom import MiniSom  # You may need to install this package (pip install minisom)
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

# Adjust the SOM parameters based on your data and requirements
som_rows = 10
som_columns = 10
som_epochs = 100
som_learning_rate = 0.1

# Initialize the SOM
som = MiniSom(som_rows, som_columns, Xnew.shape[1], sigma=1.0, learning_rate=som_learning_rate)

# Train the SOM with the transformed data
som.train(Xnew, som_epochs, verbose=True)

# Get the SOM cluster labels for each data point
som_labels = som.labels_map(Xnew)

# Convert the SOM labels to a flat array
som_flat_labels = []
for i in range(len(Xnew)):
    som_flat_labels.append(som_labels[tuple(Xnew[i])])

dfcluster = pd.DataFrame(som_flat_labels, columns=['cluster'])
dfall = pd.concat([dfpca, dfcluster], axis=1)

print(confusion_matrix(dfall['predictions'], dfall["cluster"]))
print(metrics.calinski_harabasz_score(Xnew, dfall['cluster']))
print(metrics.silhouette_score(Xnew, dfall['cluster'], metric='euclidean'))

# Plot the SOM clusters
for i in range(som_rows):
    for j in range(som_columns):
        cluster_df = dfall[dfall['cluster'] == (i, j)]
        plt.plot(cluster_df['PC1'], cluster_df['PC2'], 'o', label=f'SOM Cluster {i}-{j}')

plt.xlabel(dfpca.columns[0])
plt.ylabel(dfpca.columns[1])
plt.title('SOM Clustering')
plt.legend()
plt.show()

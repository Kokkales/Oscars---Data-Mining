import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler

TRAIN_PATH_PROCESSED = "./Data/moviesUpdated_processed.xlsx"
FULL_PREDICTIONS = './Data/full_predictions.xlsx'

def separateData(ds):
    if 'oscar winners' not in ds.columns:
        raise ValueError('oscar winners not in the dataset')
    target = ds['oscar winners']
    data = ds.drop(columns='oscar winners')
    return target, data

def get_rows_with_prediction_one(df):
    if 'oscar winners' not in df.columns:
        raise ValueError('oscar winners not in the dataframe')

    prediction_one_ids = (df[df['oscar winners'] == 1].index).tolist()
    return prediction_one_ids

if len(sys.argv)!=3:
    print("Usage: python3 clustering.py <algorithm> <cluster number> <ss/rs/mm>")
    sys.exit(1)

iris = pd.read_excel(TRAIN_PATH_PROCESSED, sheet_name='Sheet1')
y, X = separateData(iris)
prediction_one_ids = get_rows_with_prediction_one(iris)

if sys.argv[2]=='ss':
    scaler = StandardScaler()
elif sys.argv[2]=='rs':
    scaler=RobustScaler()
else:
    scaler=MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Perform PCA on the scaled data
pca = PCA(n_components=2)
Xnew = pca.fit_transform(X_scaled)
feature_names_pc1 = X.columns[np.argsort(pca.components_[0])[::-1]].tolist()

# Perform DBSCAN clustering on the scaled data
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan_labels = dbscan.fit_predict(Xnew)

# Add cluster labels to your dataframe
dfcluster_dbscan = pd.DataFrame(dbscan_labels, columns=['dbscan_cluster'])
dfpca = pd.DataFrame(Xnew, columns=["PC1", "PC2"])
dfclass = pd.DataFrame(y, columns=['oscar winners'])
dfall_dbscan = pd.concat([dfpca, dfclass, dfcluster_dbscan], axis=1)

# Print clustering stats
print("Confusion Matrix:")
print(confusion_matrix(dfall_dbscan['oscar winners'], dfall_dbscan['dbscan_cluster']))
print("Calinski-Harabasz Score:", metrics.calinski_harabasz_score(Xnew, dfall_dbscan['dbscan_cluster']))
print("Silhouette Score:", silhouette_score(Xnew, dfall_dbscan['dbscan_cluster'], metric='euclidean'))

# Explore cluster characteristics (core sample indices and labels)
core_samples_mask = np.zeros_like(dbscan.labels_, dtype=bool)
core_samples_mask[dbscan.core_sample_indices_] = True

# Loop through the DBSCAN clusters and print information for points with prediction equal to 1
pc1_osc=[]
pc2_osc=[]
for index, row in dfall_dbscan.iterrows():
    point_id = index
    pc1_value = row['PC1']
    pc2_value = row['PC2']
    cluster_label = row['dbscan_cluster']
    if point_id in prediction_one_ids:
        pc1_osc.append(pc1_value)
        pc2_osc.append(pc2_value)
        # print(f"Point ID: {point_id+1}, PC1: {pc1_value}, PC2: {pc2_value}, Cluster: {cluster_label}")

# Plot clusters according to principal components 1 and 2
fig, axs = plt.subplots(1, 2, figsize=(10, 5))

# Plot the silhouette score comparing to Fowlkes-Mallows score
xs = range(2, 50)
sils = []
fms = []

for i in xs:
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    dbscan_labels = dbscan.fit_predict(Xnew)
    sils.append(silhouette_score(Xnew, dbscan_labels, metric='euclidean'))
    fms.append(metrics.fowlkes_mallows_score(dfall_dbscan['oscar winners'], dbscan_labels))
axs[0].plot(xs, sils)
axs[0].plot(xs, fms)
axs[0].set_xlabel('Number of clusters (k)')
axs[0].set_ylabel('Score')
axs[0].set_title('Silhouette and Fowlkes-Mallows Scores')
axs[0].legend(['Silhouette', 'Fowlkes-Mallows'])

# Plot clusters according to principal components 1 and 2
for label in set(dbscan_labels):
    cluster_df = dfall_dbscan[dfall_dbscan['dbscan_cluster'] == label]
    axs[1].scatter(cluster_df['PC1'], cluster_df['PC2'], label=f'Cluster {label}')
axs[1].scatter(pc1_osc, pc2_osc, label=f'Oscars', color='red')
axs[1].set_xlabel("PC1")
axs[1].set_ylabel("PC2")
axs[1].set_title("DBSCAN Clusters in PC1-PC2 Space")
axs[1].legend()

plt.tight_layout()
plt.show()
plt.close()

from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd
def seperateData(ds):
  target=ds['oscar winners']
  col=[]
  for feature in ds.columns:
    if feature!='oscar winners':
      col.append(feature)
  data=ds[col]
  return (target,data)
# Load the iris dataset (just for testing the PCA plot)
iris = datasets.load_iris()

# Load your dataset
dataset = pd.read_excel('./datesFour.xlsx', sheet_name='Sheet1')
target, data = seperateData(dataset)

# Seprate train and test data
X = data
y = target

# Plotting in 3D using PCA
fig = plt.figure(1, figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d", elev=-150, azim=110)

# Apply PCA and plot the first three components
X_reduced = PCA(n_components=3).fit_transform(X)
scatter = ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=y, s=40)

ax.set_title("First three PCA dimensions")
ax.set_xlabel("1st Eigenvector")
ax.set_ylabel("2nd Eigenvector")
ax.set_zlabel("3rd Eigenvector")

# Adding legend
legend = ax.legend(
    *scatter.legend_elements(), loc="lower right", title="Classes"
)

plt.show()

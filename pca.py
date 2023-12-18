from dataPreprocessing import DataPreprocessor
from classifications import Classificationer
import seaborn as sns
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import math
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Generic
import pandas as pd
import numpy as np

# Generate data sets
from sklearn.datasets import make_blobs

# PCA
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Data Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

def seperateData(ds):
  target=ds['oscar winners']
  col=[]
  for feature in ds.columns:
    if feature!='oscar winners':
      col.append(feature)
  data=ds[col]
  return (target,data)


# dp=DataPreprocessor("Book.xlsx","datesThree.xlsx")
# dataset=dp.executePreprocess('normalization')#options: normalization, scaling
dataset=pd.read_excel('./datesFour.xlsx', sheet_name = 'Sheet1')
target,data=seperateData(dataset)

# # Seprate train and test data
X=data
# y=LabelEncoder().fit_transform(target)
y=target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_standardized = (X - X.mean()) / X.std()

# Apply PCA
pca = PCA()
X_pca = pca.fit_transform(X_standardized)

# Explained variance ratio
explained_variance_ratio = pca.explained_variance_ratio_

# Cumulative explained variance
cumulative_explained_variance = explained_variance_ratio.cumsum()

# Determine the number of components to keep (e.g., 95% of variance)
n_components = (cumulative_explained_variance < 0.95).sum() + 1

# Retain only the selected components
X_reduced = X_pca[:, :n_components]
print(X_reduced)

# Visualize explained variance
import matplotlib.pyplot as plt

plt.plot(cumulative_explained_variance, marker='o')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.show()

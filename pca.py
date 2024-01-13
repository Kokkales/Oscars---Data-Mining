import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Assuming 'file' is your DataFrame
file = pd.read_excel('./test_sample.xlsx', sheet_name='Sheet1')

def separateData(ds):
    target = ds['oscar winners']
    features = ds.drop(columns=['oscar winners'])
    return target, features

# Separate target and features
target, features = separateData(file)

# Standardize the data
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Apply PCA
pca = PCA()
principal_components = pca.fit_transform(scaled_features)

# Analyze explained variance
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_variance_ratio = explained_variance_ratio.cumsum()

# Determine the number of components to retain based on the explained variance threshold
threshold_variance = 0.95
num_components_to_retain = (cumulative_variance_ratio <= threshold_variance).sum()

# Get the names of features with low importance
low_importance_original_features = features.columns[-num_components_to_retain:]

# Print the names of features with low importance
print("Features with Low Importance:")
print(low_importance_original_features)

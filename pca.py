import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import pandas as pd

# Load data from Excel file
X = pd.read_excel('./trainSet.xlsx', sheet_name='Sheet1')
X = X.drop(columns='oscar winners')

# Assuming X is your feature matrix
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Scikit-learn PCA
pca = PCA(n_components=26)  # Specify the desired number of components (k)
X_pca_sklearn = pca.fit_transform(X_scaled)

# Print the results
print("Original DataFrame:")
print(X.head())

print("\nScaled Data using MinMaxScaler:")
print(pd.DataFrame(data=X_scaled, columns=X.columns).head())

print("\nScikit-learn PCA Results with MinMax Scaling:")
result_sklearn = pd.DataFrame(data=X_pca_sklearn, columns=[f'PC{i+1}' for i in range(pca.n_components_)])
print(result_sklearn.head())

# Print feature names corresponding to principal components
print("\nFeature Names for Principal Components:")
for i in range(pca.n_components_):
    feature_names = X.columns[np.argsort(np.abs(pca.components_[i]))[::-1]]
    print(f"PC{i+1}:", feature_names[:10])  # Print the top 5 features for each principal component

# Calculate cumulative explained variance ratio
cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)

# Print the cumulative explained variance ratio
print("\nCumulative Explained Variance Ratio:")
print(cumulative_variance_ratio)

# Determine the number of components to keep (e.g., 95% of variance)
desired_variance_ratio = 0.95
num_components_to_keep = np.argmax(cumulative_variance_ratio >= desired_variance_ratio) + 1

# Print the recommended number of components
print(f"\nRecommended Number of Components to Retain {desired_variance_ratio * 100}% Variance: {num_components_to_keep}")

# Get the final feature names based on the recommended number of components
final_feature_names = []
for i in range(num_components_to_keep):
    final_feature_names.extend(X.columns[np.argsort(np.abs(pca.components_[i]))[::-1]][:10])

# Remove duplicates (in case some features appear in multiple components)
final_feature_names = list(set(final_feature_names))

# Print the final set of feature names
print("\nFinal Feature Names to Keep:")
print(final_feature_names)

# Get the list of features to drop
features_to_drop = list(set(X.columns) - set(final_feature_names))

# Print the list of features to drop
print("\nFeatures to Drop:")
print(features_to_drop)

# Display the importance of each feature in the final set
print("\nImportance of Each Feature:")
for feature in final_feature_names:
    feature_importance = sum([np.abs(pca.components_[i][X.columns.get_loc(feature)]) for i in range(num_components_to_keep)])
    print(f"{feature}: {feature_importance}")

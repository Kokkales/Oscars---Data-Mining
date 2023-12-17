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

def seperateData(ds):
  target=ds['oscar winners']
  col=[]
  for feature in ds.columns:
    if feature!='oscar winners':
      col.append(feature)
  data=ds[col]
  return (target,data)


# preprocessing
# dp=DataPreprocessor("Book.xlsx","datesThree.xlsx")
# dataset=dp.executePreprocess('normalization')#options: normalization, scaling
dataset=pd.read_excel('./datesFour.xlsx', sheet_name = 'Sheet1')
target,data=seperateData(dataset)
# # ploting testing code
# # Assuming dataset is your DataFrame
# selected_features = ['average audience', 'average critics', 'opening weekend', 'domestic gross', 'foreign gross', 'budget ($million)', 'budget recovered', 'budget recovered opening weekend', 'imdb rating', 'oscar winners']

# # Calculate the number of rows and columns for the grid
# # num_features = len(selected_features)
# # num_cols = 3
# # num_rows = math.ceil(num_features / num_cols)

# # # Iterate through pairs of features
# # for i in range(len(selected_features) - 1):
# #     for j in range(i + 1, len(selected_features)):
# #         feature1 = selected_features[i]
# #         feature2 = selected_features[j]

# #         # Create a scatter plot for each pair
# #         plt.scatter(dataset[feature1], dataset[feature2], c=dataset['oscar winners'])
# #         plt.xlabel(feature1)
# #         plt.ylabel(feature2)
# #         plt.title(f'Scatter Plot: {feature1} vs {feature2}')

# #         # Adjust the layout to create a grid of scatter plots within each window
# #         plt.subplots_adjust(wspace=0.5, hspace=0.5)

# #         # Display 9 figures in each window
# #         if (i % 9 == 8) or (i == len(selected_features) - 2):
# #             plt.show()


# # dfsub=dataset[['average audience','average critics','opening weekend','domestic gross','foreign gross','budget ($million)','budget recovered','budget recovered opening weekend','imdb rating','oscar winners']]
# # sns.pairplot(dfsub,hue='oscar winners',height=10)
# # plt.show()

# # Seprate train and test data
X=data
y=LabelEncoder().fit_transform(target)


# # plt.scatter(dataset['average critics'],dataset['average audience'],c=dataset['oscar winners'])
# # plt.show()
# # plt.close()
# # classification

for i in range(1):
  X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)
  cl=Classificationer(X_train, X_test, y_train, y_test)
  # cl=Classificationer()
  # print(cl.excecuteDtcClassification())#0.96
  print(cl.executeDtrClassification()) #0.96
  # print(cl.executeRfClassification())# 0.98
  # print(cl.executeKnnClassification()) # 0.77
  # print(cl.executeGpClassification()) # 0.95
  # print(cl.executeSvmClassification()) #0.95
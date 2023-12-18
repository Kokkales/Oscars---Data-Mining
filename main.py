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


# dp=DataPreprocessor("Book.xlsx","datesThree.xlsx")
# dataset=dp.executePreprocess('normalization')#options: normalization, scaling
dataset=pd.read_excel('./datesFour.xlsx', sheet_name = 'Sheet1')
target,data=seperateData(dataset)

# # Seprate train and test data
X=data
y=LabelEncoder().fit_transform(target)

for i in range(1):
  X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)
  cl=Classificationer(X_train, X_test, y_train, y_test)
  # cl=Classificationer()
  print(cl.excecuteDtcClassification())#0.96
  print(cl.executeDtrClassification()) #0.96
  print(cl.executeRfClassification())# 0.98
  print(cl.executeKnnClassification()) # 0.77
  print(cl.executeGpClassification()) # 0.95
  print(cl.executeSvmClassification()) #0.95
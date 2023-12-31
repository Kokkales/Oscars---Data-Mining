from dataPreprocessing import DataPreprocessor
from classifications import Classificationer
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
import seaborn as sns
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
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

def fixPredictionFile(train_file,test_file):
  diff_train_columns = set(train_file.columns) - set(test_file.columns)
  diff_test_columns = set(test_file.columns) - set(train_file.columns)

  for test_column in diff_test_columns:
      for train_column in diff_train_columns:
          # Check if the columns share at least four letters
          if len(set(test_column).intersection(train_column)) >= 4:
              # Replace the column name in test_file
              test_file = test_file.rename(columns={test_column: train_column})
              break  # Stop searching for similar columns once a match is found

  # Add missing columns to test_file with values set to 0
  for train_column in diff_train_columns:
      test_file[train_column] = 0
  test_file = test_file.loc[:, ~test_file.columns.duplicated(keep='first')]
  test_file = test_file.drop(columns= set(test_file.columns) - set(train_file.columns))

  test_file.to_excel('to_predict_final.xlsx')
  # print(train_file.shape,test_file.shape)
  return test_file.sort_index(axis=1)

# def trainModel():

#   return x,y


# dp=DataPreprocessor("Book.xlsx","trained_final.xlsx")
# train_dataset=dp.executePreprocess()#options: normalization, scaling
train_dataset=pd.read_excel('trained_final.xlsx', sheet_name = 'Sheet1') #dates four give s better results
train_target,train_data=seperateData(train_dataset)
train_data=train_data.sort_index(axis=1)
# --------------------------------------------------------------
# df=DataPreprocessor("./movies_test _anon_sample.xlsx","to_predict_final.xlsx")
# test_dataset=df.executePreprocess(deleteDuplicateNames=False)#options: normalization, scaling
prediction_dataset=pd.read_excel('to_predict_final.xlsx', sheet_name = 'Sheet1') #dates four give s better results
prediction_data=fixPredictionFile(train_data,prediction_dataset)
# print(test_data.shape,train_data.shape)


imputer = SimpleImputer(strategy='mean')
if train_data.shape[1]==prediction_data.shape[1]:
  # Split and train data
  X_train, X_test, y_train, y_test = train_test_split(train_data,train_target, test_size=0.2, random_state=42)
  # Create the classifications object
  cl=Classificationer(X_train,X_test,y_train, y_test)
  # Call the classification model
  predictions=cl.executeDtrClassification()
  print(predictions)
  # # print(cl.excecuteDtcClassification())#0.96 NOt suitable (ill defined)
  # print(cl.executeDtrClassification()) #0.97
  # print(cl.executeRfClassification())# 0.97
  # # print(cl.executeKnnClassification()) # 0.77 NOT THE BEST results
  # # print(cl.executeGpClassification()) # 0.95 NOT the best results
  # # print(cl.executeSvmClassification()) #0.95 NOT suitable (ill defined)

  # print the accuracy line









  # # # Now, load the data for prediction (to_predict_final.xlsx)
  # to_predict_data = prediction_data

  # # # Handle missing values in the prediction data
  # to_predict_data = pd.DataFrame(imputer.transform(to_predict_data), columns=to_predict_data.columns)

  # # # Make predictions on the data for prediction
  # upredictions = model.predict(to_predict_data)

  # # # Add the predictions to the original dataframe if needed
  # # to_predict_data['Predicted_Oscar_Winners'] = predictions

  # # # Save the results to a new file or use them as needed
  # # to_predict_data.to_excel('predictionsTwo.xlsx', index=False)






















  # print("-----------------CLASSIFICATION TRAINING------------------")
  # # TRAIN THE MODEL
  # # # Seprate train and test data
  # X=train_data
  # y=LabelEncoder().fit_transform(train_target)

  # # PREDICT WITH THE NEW SET OF DATA
  # scaler = StandardScaler()
  # X_train = scaler.fit_transform(X)
  # # X_test = scaler.transform(X_test)
  # print("-----------------CLASSIFICATION PREDICTING------------------")

  # for i in range(20):
  # X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)
  # cl=Classificationer(X_train, X_test, y_train, y_test)
  # # cl=Classificationer()
  # # print(cl.excecuteDtcClassification())#0.96 NOt suitable (ill defined)
  # print(cl.executeDtrClassification()) #0.97
  # print(cl.executeRfClassification())# 0.97
  # # print(cl.executeKnnClassification()) # 0.77 NOT THE BEST results
  # # print(cl.executeGpClassification()) # 0.95 NOT the best results
  # # print(cl.executeSvmClassification()) #0.95 NOT suitable (ill defined)
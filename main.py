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
from sklearn.metrics import confusion_matrix, classification_report
import math

def seperateData(ds):
  target=ds['oscar winners']
  col=[]
  for feature in ds.columns:
    if feature!='oscar winners':
      col.append(feature)
  data=ds[col]
  return (target,data)

def fixTestFile(train_file,test_file):
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

  test_file.to_excel('edited_final.xlsx')
  # print(train_file.shape,test_file.shape)
  return test_file.sort_index(axis=1)


# dp=DataPreprocessor("Book.xlsx","trained_final.xlsx")
# train_dataset=dp.executePreprocess()#options: normalization, scaling
train_dataset=pd.read_excel('test.xlsx', sheet_name = 'Sheet1') #dates four give s better results
train_target,train_data=seperateData(train_dataset)
train_data=train_data.sort_index(axis=1)
# --------------------------------------------------------------
# df=DataPreprocessor("./movies_test _anon_sample.xlsx","to_predict_final.xlsx")
# test_dataset=df.executePreprocess(deleteDuplicateNames=False)#options: normalization, scaling
test_dataset=pd.read_excel('to_predict_final.xlsx', sheet_name = 'Sheet1') #dates four give s better results
test_data=fixTestFile(train_data,test_dataset)
# print(test_data.shape,train_data.shape)


imputer = SimpleImputer(strategy='mean')
if train_data.shape[1]==test_data.shape[1]:
  X_train, X_valid, y_train, y_valid = train_test_split(train_data,train_target, test_size=0.2,random_state=42)

  # Train a machine learning model (e.g., RandomForestClassifier)
  # model = RandomForestClassifier()
  # model.fit(X_train, y_train)
  model = DecisionTreeClassifier()
  model.fit(X_train, y_train)

  # Make predictions on the validation set
  y_pred = model.predict(X_valid)

  # Evaluate the model on the validation set
  accuracy = accuracy_score(y_valid, y_pred)

  conf_matrix = confusion_matrix(y_valid, y_pred)
  print("Confusion Matrix:\n", conf_matrix)
  class_report = classification_report(y_valid, y_pred)
  print("Classification Report:\n", class_report)
  print(f'Accuracy on the validation set: {accuracy}')

  # Now, load the data for prediction (to_predict_final.xlsx)
  to_predict_data = test_data

  # Handle missing values in the prediction data
  # to_predict_data = pd.DataFrame(imputer.transform(to_predict_data), columns=to_predict_data.columns)

  # Make predictions on the data for prediction
  predictions = model.predict(to_predict_data)

  # Add the predictions to the original dataframe if needed
  to_predict_data['Predicted_Oscar_Winners'] = predictions

  # Save the results to a new file or use them as needed
  to_predict_data.to_excel('predictionsFour.xlsx', index=False)






















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
from dataPreprocessing import DataPreprocessor
from classifications import Classificationer
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import recall_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingClassifier
import math
import subprocess
from xgboost import XGBClassifier

def seperateData(ds):
  if 'oscar winners' not in ds.columns:
    raise ValueError('Oscar winners not in the dataset')
  target=ds['oscar winners']
  col=[]
  data=ds.drop(columns='oscar winners')
  # for feature in ds.columns:
  #   if feature!='oscar winners':
  #     col.append(feature)
  # data=ds[col]
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


scaler = MinMaxScaler()
# dp=DataPreprocessor("Book.xlsx","trained_final.xlsx")
# trainDataset=dp.executePreprocess()#options: normalization, scaling
trainDataset=pd.read_excel('test.xlsx', sheet_name = 'Sheet1') #dates four give s better results
datasetColumns=trainDataset.columns
trainTarget,trainData=seperateData(trainDataset)
trainData=trainData.sort_index(axis=1)
scaledTrainDataset=scaler.fit_transform(trainData)
# --------------------------------------------------------------
# df=DataPreprocessor("./movies_test _anon_sample.xlsx","to_predict_final.xlsx")
# predictDataset=df.executePreprocess(deleteDuplicateNames=False)#options: normalization, scaling
predictDataset=pd.read_excel('to_predict_final.xlsx', sheet_name = 'Sheet1') #dates four give s better results
predictData=fixTestFile(trainData,predictDataset)
scaledPredictData=scaler.transform(predictData)

imputer = SimpleImputer(strategy='mean')
if trainData.shape[1]==predictData.shape[1]:
  sum=0
  for i in range(1):
    X_train, X_valid, y_train, y_valid = train_test_split(trainData,trainTarget, test_size=0.25,random_state=42)

    # Train a machine learning model (e.g., RandomForestClassifier)
    # model = RandomForestClassifier(random_state=42) #recall 0.17, f1-score 0.29 acc=0.94 ill defined
    # model = LogisticRegression(max_iter=1500, random_state=42)
    model = DecisionTreeRegressor(random_state=42) #recall 0.28, f1-score 0.34, acc=0.92
    # model = GradientBoostingClassifier(random_state=42)

    # model = XGBClassifier(random_state=42)
    # model = DecisionTreeClassifier(random_state=42)#recall 0.28, f1-score 0.33, acc=0.92
    # model= MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42) #ill defined
    # model=SVC()# ill defined
    # model=KNeighborsClassifier(n_neighbors=3) # ill defined
    # model=LogisticRegression(random_state=42) # ill defined
    # model=GaussianNB() # ill defined
    # model= SGDClassifier(loss='hinge', alpha=0.0001, max_iter=1000, random_state=42) # ill defined

    model.fit(X_train, y_train)

    # --------------------------------------------------------------------------------------------
    feature_importances = model.feature_importances_

    # Create a DataFrame to display features and their importance scores
    feature_importance_df = pd.DataFrame({'Feature': trainData.columns, 'Importance': feature_importances})

    # Sort the DataFrame by importance scores in descending order
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    # Select the top 10 features
    top_10_features = feature_importance_df.head

    # Display the top 10 features
    print(top_10_features)
    # --------------------------------------------------------------------------------------------
    # Make predictions on the validation set
    y_pred = model.predict(X_valid)

    # Evaluate the model on the validation set
    accuracy = accuracy_score(y_valid, y_pred)

    conf_matrix = confusion_matrix(y_valid, y_pred)
    print("Confusion Matrix:\n", conf_matrix)
    class_report = classification_report(y_valid, y_pred)
    print("Classification Report:\n", class_report)
    print(f'Accuracy on the validation set: {accuracy}')
    class_rep = classification_report(y_valid, y_pred)

    sum =sum+ recall_score(y_valid, y_pred)

  print(f"Average Recall: {sum/1}")
  predictions = model.predict(predictData)
  predictData['predictions'] = predictions
  predictData['id'] = range(1, len(predictData) + 1)
  results=predictData[['id','predictions']]
  results.to_excel('predictionsFour.xlsx', index=False)
  results.to_csv('predictionsFour.csv', index=False)
  # print(results.head())
  subprocess.Popen(['start','excel','predictionsFour.xlsx'],shell=True)























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
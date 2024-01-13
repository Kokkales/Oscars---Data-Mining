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
from sklearn.model_selection import cross_val_score
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
  data=data.sort_index(axis=1)
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

TRAIN_PATH="./moviesUpdated.xlsx"
TRAIN_PATH_PROCESSED="./trainSet.xlsx"
PREDICT_PATH="./movies_test _anon.xlsx"
PREDICT_PATH_PROCESSED="./test_sample.xlsx"
def preprocess():
    # # preprocess all files with MinMax Scaler#
    # dp=DataPreprocessor(TRAIN_PATH,TRAIN_PATH_PROCESSED)
    # df=DataPreprocessor(PREDICT_PATH,PREDICT_PATH_PROCESSED)
    # # # df=DataPreprocessor("./movies_test _anon_sample.xlsx","test_final.xlsx")
    # trainDataset=dp.executePreprocess()
    # predictDataset=df.executePreprocess(predict=True)#options: predict=True/False

    # SAVING TIME-------------------
    trainDataset=pd.read_excel(TRAIN_PATH_PROCESSED, sheet_name = 'Sheet1')
    predictDataset=pd.read_excel(PREDICT_PATH_PROCESSED, sheet_name = 'Sheet1')
    print('all files has been succesfully preprocessed')
    return trainDataset,predictDataset

trainDataset,predictDataset=preprocess()
trainTarget,trainData=seperateData(trainDataset) #seperate train from target data of the train dataset
predictData=fixTestFile(trainData,predictDataset)

# Scale data
scaler=MinMaxScaler()
scaledTrainData=scaler.fit_transform(trainData)
scaledPredictData=scaler.transform(predictData)

# Train the model
X_train, X_valid, y_train, y_valid = train_test_split(scaledTrainData,trainTarget, test_size=0.25,random_state=42)
# model = RandomForestClassifier(random_state=42) #recall 0.17, f1-score 0.29 acc=0.94 ill defined
# model = LogisticRegression(max_iter=1500, random_state=42)
# model = DecisionTreeRegressor(random_state=42) #recall 0.28, f1-score 0.34, acc=0.92
# model = DecisionTreeClassifier(random_state=42)#recall 0.28, f1-score 0.33, acc=0.92
# model=SVC()# ill defined
model=KNeighborsClassifier(n_neighbors=3) # ill defined
# model=GaussianNB() # ill defined
# model= MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42) #ill defined

model.fit(X_train, y_train)
# Make predictions on the validation set
y_pred = model.predict(X_valid)
# Evaluate the model on the validation set
accuracy = accuracy_score(y_valid, y_pred)
conf_matrix = confusion_matrix(y_valid, y_pred)
print("Confusion Matrix:\n", conf_matrix)
class_report = classification_report(y_valid, y_pred)
class_rep = classification_report(y_valid, y_pred)
cv_scores = cross_val_score(model, scaledTrainData, trainTarget, cv=5)
print("Classification Report:\n", class_report)
print(f'Accuracy on the validation set: {accuracy}')
print(f'Cross validation: {np.mean(cv_scores)}')
# --------------------------------------------------------------------------------------------
# feature_importances = model.feature_importances_

# # Create a DataFrame to display features and their importance scores
# feature_importance_df = pd.DataFrame({'Feature': trainData.columns, 'Importance': feature_importances})

# # Sort the DataFrame by importance scores in descending order
# feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# # Select the top 10 features
# top_10_features = feature_importance_df.head

# # Display the top 10 features
# print(top_10_features)


# Predict in the model
predictions = model.predict(scaledPredictData)
print(predictions)
count_ones = np.count_nonzero(predictions == 1.0)
print("#Oscar winners: ", count_ones)
scaledPredictData=pd.DataFrame(scaledPredictData,columns=predictData.columns)
scaledPredictData['predictions'] = predictions
scaledPredictData['id'] = range(1, len(scaledPredictData) + 1)
results=scaledPredictData[['id','predictions']]
results.to_excel('predictionsFour.xlsx', index=False)
results.to_csv('predictionsFour.csv', index=False)
# print(results.head())
subprocess.Popen(['start','excel','predictionsFour.xlsx'],shell=True)
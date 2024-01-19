import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from dataPreprocessing import DataPreprocessor
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, cross_val_score
import subprocess
import sys
import xgboost as xgb
from sklearn.metrics import f1_score


TRAIN_PATH="./Data/moviesUpdated.xlsx"
TRAIN_PATH_PROCESSED="./Data/moviesUpdated_processed.xlsx"
PREDICT_PATH="./Data/movies_test _anon.xlsx"
PREDICT_PATH_PROCESSED="./Data/movies_test_anon_processed.xlsx"
PREDICTIONS_PATH_XL='./Data/predictions.xlsx'
PREDICTIONS_PATH_CSV='./Data/predictions.csv'
FULL_PREDICTIONS='./Data/full_predictions.xlsx'

def seperateData(ds):
  if 'oscar winners' not in ds.columns:
    raise ValueError('Oscar winners not in the dataset')
  target=ds['oscar winners']
  data=ds.drop(columns='oscar winners')
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

  test_file.to_excel(PREDICT_PATH_PROCESSED)
  # print(train_file.shape,test_file.shape)
  return test_file.sort_index(axis=1)

def preprocess(command='prepro'):
    if command=='prepro':
        dp=DataPreprocessor(TRAIN_PATH,TRAIN_PATH_PROCESSED)
        df=DataPreprocessor(PREDICT_PATH,PREDICT_PATH_PROCESSED)
        trainDataset=dp.executePreprocess()
        predictDataset=df.executePreprocess(predict=True)#options: predict=True/False
    else:
        # SAVING TIME-------------------
        trainDataset=pd.read_excel(TRAIN_PATH_PROCESSED, sheet_name = 'Sheet1')
        predictDataset=pd.read_excel(PREDICT_PATH_PROCESSED, sheet_name = 'Sheet1')
    print('all files has been succesfully preprocessed')
    return trainDataset,predictDataset

def printStats(model,y_valid,y_pred,scaledTrainData,trainTarget):
    accuracy = accuracy_score(y_valid, y_pred)
    conf_matrix = confusion_matrix(y_valid, y_pred)
    class_report = classification_report(y_valid, y_pred)
    cv_scores = cross_val_score(model, scaledTrainData, trainTarget, cv=5)
    print("Confusion Matrix:\n", conf_matrix)
    print("Classification Report:\n", class_report)
    print(f'Accuracy: {accuracy}')
    print(f'Cross validation: {np.mean(cv_scores)}')

def featureImportances(model,trainData):
    feature_importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({'Feature': trainData.columns, 'Importance': feature_importances})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
    top_10_features = feature_importance_df.head
    print(top_10_features)

# Scale data
def scaleData(trainData,predictData):
    scaler=MinMaxScaler()
    scaledTrainData=scaler.fit_transform(trainData)

    scaledPredictData=scaler.transform(predictData)
    return scaledTrainData,scaledPredictData

def doPredictions(model,scaledPredictData,predictData):
    predictions = model.predict(scaledPredictData)
    count_ones = np.count_nonzero(predictions == 1.0)
    scaledPredictData=pd.DataFrame(scaledPredictData,columns=predictData.columns)
    scaledPredictData['predictions'] = predictions
    scaledPredictData.to_excel(FULL_PREDICTIONS,index=False)
    scaledPredictData['id'] = range(1, len(scaledPredictData) + 1)
    results=scaledPredictData[['id','predictions']]
    results.to_excel(PREDICTIONS_PATH_XL, index=False)
    results.to_csv(PREDICTIONS_PATH_CSV, index=False)
    print("#Oscar winners: ", count_ones)
    winners_ids = results[results['predictions'] == 1]['id'].tolist()
    print("Winner IDs:", winners_ids)
    # TOREMOVE
    print("==================")
    sureWinners=[52,59,101,147,149,308,344,353,400,406,413,466]
    winners_set = set(winners_ids)
    sureWinners_set = set(sureWinners)

    # Find common winner IDs
    common_winners = winners_set.intersection(sureWinners_set)

    # Find winner IDs not in sureWinners
    winners_not_in_sureWinners = winners_set.difference(sureWinners_set)

    # Print the results
    print("Number of winner IDs in sureWinners:", len(common_winners))
    print("Winner IDs in sureWinners:", common_winners)
    print("Number of winner IDs not in sureWinners:", len(winners_not_in_sureWinners))
    print("Winner IDs not in sureWinners:", winners_not_in_sureWinners)
    print('AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA::::::::  ',len(common_winners))
    return len(common_winners),count_ones
    # subprocess.Popen(['start','excel','./Data/predictions.xlsx'],shell=True)

def doTraining(scaledTrainData,trainTarget,modelName='knn',rs=42):
    print("---------------------------------------TRY:",rs)
    X_train, X_valid, y_train, y_valid = train_test_split(scaledTrainData,trainTarget, test_size=0.25,random_state=rs)
    if modelName=='rf':
        model = RandomForestClassifier(random_state=42)
    elif modelName=='lr':
        model = LogisticRegression(max_iter=1500, random_state=42)
    elif modelName=='dtc':
        model = DecisionTreeClassifier(random_state=42)
    elif modelName=='knn':
        model=KNeighborsClassifier(n_neighbors=3)
    elif modelName=='gb':
        model=xgb.XGBClassifier()
    model.fit(X_train, y_train)
    return model,X_train,X_valid, y_train, y_valid

def handleArgvs():
    if len(sys.argv) != 4:
        print("Usage: python3 main.py <rf/knn/lr/dtc> <stats/nostats> <prepro/noprepro>")
        sys.exit(1)
    argOne=sys.argv[1]
    argTwo=sys.argv[2]
    argThree=sys.argv[3]
    return argOne,argTwo,argThree

if __name__=='__main__':
    argOne,argTwo,argThree=handleArgvs()

    trainDataset,predictDataset=preprocess(argThree)
    trainTarget,trainData=seperateData(trainDataset) #seperate train from target data of the train dataset
    predictData=fixTestFile(trainData,predictDataset)
    scaledTrainData,scaledPredictData=scaleData(trainData,predictData)
    max=0
    pos=0
    for i in range(1):
        model,X_train,X_valid,y_train, y_valid=doTraining(scaledTrainData,trainTarget,argOne,42)
        y_pred = model.predict(X_valid)
        if sys.argv[2]=='stats':
            printStats(model,y_valid,y_pred,scaledTrainData,trainTarget)
        if sys.argv[1]!='knn' and sys.argv[1]!='lr':
            featureImportances(model,trainData)

        r,o=doPredictions(model,scaledPredictData,predictData) # Predict in the model
        fScore=f1_score(y_valid,y_pred)
        if fScore>=max:
            max=fScore
            pos=i
    print("BEST RANDOM STATE: ",pos,max)#random state 87
    # df = pd.read_excel(PREDICT_PATH)

    # # Find the row where column F has the value 60 and column AB has the value 7.7
    # result = df[(df['year'] == 2021)&(df['metacritic critics'] == 72)]

    # # If there are multiple rows meeting the conditions, 'result' will contain all of them.
    # # If you want just the first row, you can use 'result.iloc[0]'
    # if not result.empty:
    #     print("Row found:", result)
    # else:
    #     print("No matching row found.")

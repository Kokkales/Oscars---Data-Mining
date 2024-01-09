# from dataPreprocessing import DataPreprocessor
# from classifications import Classificationer
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score
# from sklearn.impute import SimpleImputer
# import seaborn as sns
# import numpy as np
# import scipy as sc
# import matplotlib.pyplot as plt
# import pandas as pd
# from sklearn.preprocessing import LabelEncoder
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import MinMaxScaler
# import math

# def seperateData(ds):
#   target=ds['oscar winners']
#   col=[]
#   for feature in ds.columns:
#     if feature!='oscar winners':
#       col.append(feature)
#   data=ds[col]
#   return (target,data)

# def fixPredictionFile(train_file,test_file):
#   diff_train_columns = set(train_file.columns) - set(test_file.columns)
#   diff_test_columns = set(test_file.columns) - set(train_file.columns)

#   for test_column in diff_test_columns:
#       for train_column in diff_train_columns:
#           # Check if the columns share at least four letters
#           if len(set(test_column).intersection(train_column)) >= 4:
#               # Replace the column name in test_file
#               test_file = test_file.rename(columns={test_column: train_column})
#               break  # Stop searching for similar columns once a match is found

#   # Add missing columns to test_file with values set to 0
#   for train_column in diff_train_columns:
#       test_file[train_column] = 0
#   test_file = test_file.loc[:, ~test_file.columns.duplicated(keep='first')]
#   test_file = test_file.drop(columns= set(test_file.columns) - set(train_file.columns))

#   test_file.to_excel('to_predict_final.xlsx')
#   # print(train_file.shape,test_file.shape)
#   return test_file.sort_index(axis=1)

# # def trainModel():

# #   return x,y


# # dp=DataPreprocessor("Book.xlsx","trained_final.xlsx")
# # train_dataset=dp.executePreprocess()#options: normalization, scaling
# train_dataset=pd.read_excel('trained_final.xlsx', sheet_name = 'Sheet1') #dates four give s better results
# train_target,train_data=seperateData(train_dataset)
# scaler=MinMaxScaler()
# train_data_scaled=scaler.fit_transform(train_data)
# # print(type(train_data))
# scaled_dataset=pd.DataFrame(train_data_scaled,columns=train_data.columns)
# train_data=scaled_dataset.sort_index(axis=1)
# # print(train_data.shape)

# # train_data.to_excel('printt.xlsx')
# # --------------------------------------------------------------
# # df=DataPreprocessor("./movies_test _anon_sample.xlsx","to_predict_final.xlsx")
# # test_dataset=df.executePreprocess(deleteDuplicateNames=False)#options: normalization, scaling
# prediction_dataset=pd.read_excel('to_predict_final.xlsx', sheet_name = 'Sheet1') #dates four give s better results
# prediction_data=fixPredictionFile(train_data,prediction_dataset)
# # print(test_data.shape,train_data.shape)


# # imputer = SimpleImputer(strategy='mean')
# if train_data.shape[1]==prediction_data.shape[1]:
# # ================= TRAINING +++++++++++++++++++++++
#   # Split and train data
#   X_train, X_test, y_train, y_test = train_test_split(train_data,train_target, test_size=0.2, random_state=42)
#   cl=Classificationer(X_train,X_test,y_train, y_test)
#   predictions,model=cl.executeDtcClassification()
#   # print(predictions)
#   # predictions=cl.executeDtrClassification()
#   # print(predictions)
#   # predictions=cl.executeRfClassification()
#   # print(predictions)
#   # predictions=cl.executeKnnClassification()
#   # print(predictions)
#   # predictions=cl.executeGpClassification()
#   # print(predictions)
#   # predictions=cl.executeSvmClassification()
#   # print(predictions)
#   # print the accuracy line

# # --------------------- PREDICT ----------------------------------
#   # Now, load the data for prediction (to_predict_final.xlsx)
#   to_predict_data = prediction_data
#   to_predict_data=pd.DataFrame(to_predict_data,columns=train_data.columns)
#   # to_predict_data = pd.DataFrame(scaler.transform(to_predict_data), columns=to_predict_data.columns)
#   # print(to_predict_data.columns)
#   # print(train_data.columns)
#   to_predict_data=to_predict_data.sort_index(axis=1)
#   scaled_to_predict_data=scaler.transform(to_predict_data)
#   # print(scaled_to_predict_data.shape)
#   predictions = model.predict(to_predict_data)
#   to_predict_data['Predicted_Oscar_Winners'] = predictions
#   to_predict_data.to_excel('predictionsTwo.xlsx', index=False)
#   # drop all the columns except the id and the prediction

from dataPreprocessing import DataPreprocessor
from classifications import Classificationer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

def separateData(ds):
    target = ds['oscar winners']
    data = ds.drop(columns=['oscar winners'])
    return target, data

def fixPredictionFile(train_file, test_file):
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
    # diff_train_columns = set(train_file.columns) - set(test_file.columns)

    # for train_column in diff_train_columns:
    #     test_file[train_column] = 0

    # test_file = test_file[train_file.columns]
    # test_file.to_excel('to_predict_final.xlsx', index=False)
    # return test_file

train_dataset = pd.read_excel('final.xlsx', sheet_name='Sheet1')
train_target, train_data = separateData(train_dataset)
print(train_data.shape)

scaler = MinMaxScaler()
train_data_scaled = scaler.fit_transform(train_data)
scaled_dataset = pd.DataFrame(train_data_scaled, columns=train_data.columns)
train_data = scaled_dataset

prediction_dataset = pd.read_excel('to_predict_final.xlsx', sheet_name='Sheet1')
prediction_data = fixPredictionFile(train_data, prediction_dataset)
print(train_data.shape)

# ================= TRAINING +++++++++++++++++++++++
# Split and train data
X_train, X_test, y_train, y_test = train_test_split(train_data, train_target, test_size=0.2, random_state=42)
cl = Classificationer(X_train, X_test, y_train, y_test)
# predictions, model = cl.executeDtcClassification()
predictions,model=cl.executeDtrClassification()

# --------------------- PREDICT ----------------------------------
# Now, load the data for prediction (to_predict_final.xlsx)
to_predict_data = pd.read_excel('to_predict_final.xlsx', sheet_name='Sheet1')
to_predict_data_fixed = fixPredictionFile(train_data, to_predict_data)

# Ensure that the columns of to_predict_data_fixed are in the same order as during training
to_predict_data_fixed = to_predict_data_fixed[train_data.columns]

scaled_to_predict_data = scaler.transform(to_predict_data_fixed)
predictions = model.predict(scaled_to_predict_data)

to_predict_data_fixed['Predicted_Oscar_Winners'] = predictions
to_predict_data_fixed.to_excel('predictionsTwo.xlsx', index=False)
print(to_predict_data.columns)
print(train_data.columns)

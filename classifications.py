import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import warnings
from sklearn.metrics import classification_report,confusion_matrix,precision_score,f1_score,recall_score,accuracy_score
from sklearn.exceptions import UndefinedMetricWarning
# from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors._nearest_centroid import NearestCentroid
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder

class Classificationer():

  def __init__(self,X_train, X_test, y_train, y_test):
    self.X_train=X_train
    self.X_test=X_test
    self.y_train=y_train
    self.y_test=y_test

  def printAccuracy(self,predictions):
    print(confusion_matrix(self.y_test,predictions))
    print('\n')
    print(classification_report(self.y_test,predictions))
    accuracy = accuracy_score(self.y_test, predictions)
    print(f'Accuracy on the validation set: {accuracy}')
    return accuracy


  # DECISION TREE CLASSIFIER
  def executeDtcClassification(self):
    model=DecisionTreeClassifier(criterion='entropy',min_samples_split=80)
    model.fit(self.X_train,self.y_train)
    predictions = model.predict(self.X_test)
    # print(confusion_matrix(self.y_test,predictions))
    # print('\n')
    # print(classification_report(self.y_test,predictions,zero_division=1))
    acc=self.printAccuracy(predictions)
    return f'Decision tree classifier working with acc={acc}.',model


  # DT
  def executeDtrClassification(self):
    model=DecisionTreeRegressor()
    model.fit(self.X_train,self.y_train)
    predictions = model.predict(self.X_test)
    # with warnings.catch_warnings():
    #   warnings.simplefilter("ignore", category=UndefinedMetricWarning)
    acc=self.printAccuracy(predictions)
    return f'Decision Tree Regressor working. with acc={acc} on the validation test.',model

  # RANDOM FOREST
  # warnings.filterwarnings('ignore')
  def executeRfClassification(self):
    model=RandomForestClassifier(n_estimators=10, max_depth=None,min_samples_split=2, random_state=0)
    model.fit(self.X_train,self.y_train)
    # with warnings.catch_warnings():
    #   warnings.simplefilter("ignore", category=UndefinedMetricWarning)
    predictions = model.predict(self.X_test)
    # print(model.feature_importances_)
    # print(confusion_matrix(self.y_test,predictions))
    # print('\n')
    # print(classification_report(self.y_test,predictions))
    acc=self.printAccuracy(predictions)
    importance = model.feature_importances_
    columns=self.X_train.columns
    rfc_cof=pd.Series(importance,columns)
    rfc_cof.sort_values().plot.barh()
    rcf_cof=rfc_cof.sort_values()
    rcf_cof.to_excel('stats.xlsx',header=True)
    # plt.show()
    # print(rfc_cof)
    # correct_predictions = (self.y_test == predictions)# this is yhat with ypredictions

    return f'Random Forest working with acc={acc}.'

  # K Nearest neighbours
  def executeKnnClassification(self):
    model=NearestCentroid()
    model.fit(self.X_train,self.y_train)
    # with warnings.catch_warnings():
    #   warnings.simplefilter("ignore", category=UndefinedMetricWarning)
    predictions = model.predict(self.X_test)
    # print(model.feature_importances_)
    # print(confusion_matrix(self.y_test,predictions))
    # print('\n')
    # print(classification_report(self.y_test,predictions))
    acc=self.printAccuracy(predictions)
    return f"KNN is working with ac{acc}"


  # Gausian Probabilistic
  def executeGpClassification(self):
    model=GaussianNB()
    model.fit(self.X_train,self.y_train)
    # with warnings.catch_warnings():
    #   warnings.simplefilter("ignore", category=UndefinedMetricWarning)
    predictions = model.predict(self.X_test)
    # print(model.feature_importances_)
    # print(confusion_matrix(self.y_test,predictions))
    # print('\n')
    # print(classification_report(self.y_test,predictions))
    acc=self.printAccuracy(predictions)
    return f'Gaussian Probabilistic working with acc={acc}.'
  # SVM
  def executeSvmClassification(self):
    model=SVC()
    model.fit(self.X_train,self.y_train)
    predictions = model.predict(self.X_test)
    # with warnings.catch_warnings():
    #   warnings.simplefilter("ignore", category=UndefinedMetricWarning)
    # sklearn.metrics.f1_score(self.y_test, predictions, average='weighted', labels=np.unique(predictions))
    # print(confusion_matrix(self.y_test,predictions))
    # print('\n')
    # print(classification_report(self.y_test,predictions))
    acc=self.printAccuracy(predictions)
    return f'SVM working with acc={acc}.'

  # SGD
  def executeSgdClassification(self):
    return 'SGD working.'

  # NN
  def executeNnClassification(self):
    return 'Nearest Neighbours working.'
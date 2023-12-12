import numpy as np
import pandas as pd

class Classificationer():

  def __init__(self,data=None,target=None):
    self.data=data
    self.target=target

  # SVM
  def executeSvmClassification(self):
    return 'SVM working.'

  # SGD
  def executeSgdClassification(self):
    return 'SGD working.'

  # NN
  def executeNnClassification(self):
    return 'Nearest Neighbours working.'

  # DT
  def executeDtClassification(self):
    return 'Decision Trees working.'

  # GP
  def executeGpClassification(self):
    return 'Gaussian Process working.'
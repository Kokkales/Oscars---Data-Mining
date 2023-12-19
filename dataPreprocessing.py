import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from textblob import Word
from imdb import IMDb
from openpyxl import load_workbook, Workbook


USELESS_COL=['rotten tomatoes critics','metacritic critics','rotten tomatoes audience','metacritic audience','rotten tomatoes vs metacritic deviance','audience vs critics deviance','primary genre','opening weekend ($million)','worldwide gross','worldwide gross ($million)','domestic gross ($million)','foreign gross ($million)','worldwide gross ($million)','of gross earned abroad','distributor','imdb vs rt disparity']
COMMA_COL=['average critics','average audience','opening weekend','foreign gross','domestic gross','budget ($million)','budget recovered','budget recovered opening weekend','imdb rating']
STRING_COL=['script type','genre','oscar winners','release date (us)']
# PRECENTAGES=['budget recovered','budget recovered opening weekend']
TYPES=['adaptation','original','based on a true story','sequel','remake']
SCALING_COL=['average critics','average audience','opening weekend','domestic gross','foreign gross','budget ($million)','budget recovered','budget recovered opening weekend','imdb rating'
]

class colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    END = '\033[0m'

def oneHotEncoding(file):
  # SCRIPT TYPE
  if 'script type' in file.columns:
    try:
      # one_hot_encoded = file['script type'].str.get_dummies(',').astype(int) # not working
      one_hot_encoded = pd.get_dummies(file['script type'].apply(lambda x: next((t for t in TYPES if str(x).startswith(t)), None))).astype(int)
      # Concatenate the one-hot encoded DataFrame with the original DataFrame
      file = pd.concat([file, one_hot_encoded], axis=1)
      file=dropUseless(file,['script type'])
    except:
      raise RuntimeError(f'{colors.RED}A problem occured while one-hot encoding script-type{colors.END}')

  # OSCAR DETAILS
  if 'oscar detail' in file.columns:
    try:
      one_hot_encoded = file['oscar detail'].str.get_dummies(',').astype(int)
      # Concatenate the one-hot encoded DataFrame with the original DataFrame
      file = pd.concat([file, one_hot_encoded], axis=1)
      file=dropUseless(file,['oscar detail'])
    except:
      raise RuntimeError(f'{colors.RED}A problem occured while one-hot encoding oscar details{colors.END}')

  # GENRE
  if 'genre' in file.columns:
    try:
      file['genre']=file['genre'].str.lower()
      genres=set()
      correctGenre=[]
      for item in file['genre']:
        for word in item.split():
          # correctedWords=[]
          correctedWord=str(Word(word).correct().lower())
          # print(correctedWord)
          correctGenre.append(correctedWord)
      genres.update(correctGenre)
      # print("c:",genres)
      words_to_remove = set()
      for word1 in genres:
            for word2 in genres:
                if word1 != word2 and len(word1) >3 and len(word2) > 3:
                    common_substrings = set([word1[i:i+5] for i in range(len(word1)-4) if word1[i:i+5] in word2])
                    if common_substrings:
                        shorter_word = word1 if len(word1) < len(word2) else word2
                        words_to_remove.add(shorter_word)
      genres.difference_update(words_to_remove)
      # print(genres)
      file['genre'] = file['genre'].apply(lambda cell: ' '.join(
        [next((word_set_word for word_set_word in genres if word_set_word[:3] == word[:3]), word) for word in cell.split()]
    ))
      for genre in genres:
          file[genre] = file['genre'].apply(lambda x: 1 if genre in x.split() else 0)
      file=dropUseless(file,['genre'])
    except:
      raise RuntimeError(f'{colors.RED}A problem occured while one-hot encoding genres{colors.END}')

  # DATE
  if 'release date (us)' in file.columns:
    try:
      file['release date (us)'] = pd.to_datetime(file['release date (us)'],format='mixed')
      # Extract month and day
      file['release date (us)'] = file['release date (us)'].dt.strftime('%m').astype(int)
      monthMapping = {
        1: 'january',
        2: 'february',
        3: 'march',
        4: 'april',
        5: 'may',
        6: 'june',
        7: 'july',
        8: 'august',
        9: 'september',
        10: 'october',
        11: 'november',
        12: 'december'
    }

      # Nominalize the 'release date (us)' column
      file['release date (us)'] = file['release date (us)'].map(monthMapping)
      one_hot_encoded = pd.get_dummies(file['release date (us)'], prefix='').astype(int)
      file = pd.concat([file, one_hot_encoded], axis=1)
      file=dropUseless(file,['release date (us)'])
    except:
      raise RuntimeError(f'{colors.RED}A problem occured while one-hot encoding dates{colors.END}')

  # print(file.shape)
  print(f'{colors.GREEN}ONE HOT ENCODING HAS BEEN SUCESSFULLY COMPLETED!{colors.END}')
  return file

def precentagesToDecimal(file):
  try:
    #  Convert each value of the df that is ending with the '%' to decimal (divide by 100)
    file = file.map(lambda val: float(val.rstrip('%')) / 100 if isinstance(val, str) and val.endswith('%') else val)
  except:
    raise RuntimeError(f'{colors.RED}A problem occured while converting precentages to decimal.{colors.END}')
  print(f"{colors.GREEN}PRECENTAGES HAS BEEN SUCCESFULLY CONVERTED!{colors.END}")
  return file

# def NumericalToNominal(file):
#   print(f"Numerical to Nominal")
#   return file

def dropUseless(file,uselessColumns):
  try:
    for item in uselessColumns:
      if item not in file.columns:
        uselessColumns.remove(item)
        continue
    if len(uselessColumns)!=0:
      file.drop(uselessColumns,axis=1,inplace=True) # deleting useless data from the excel
  except:
    raise RuntimeError(f'{colors.RED}A problem occured while dropping useless columns.{colors.END}')
  print(f"{colors.GREEN}USELESS COLUMNS HAS BEEN SUCCESFULLY DELETED!{colors.END}")
  return file

def deleteDuplicate(file):
  try:
    if (file.duplicated().sum() != 0) or (not file[file.duplicated(subset=['film'])].empty):
      print(f'The dataset contains {(file.duplicated(subset=["film"])).sum()} duplicate films that need to be removed.')
      print(f'The dataset contains {file.duplicated().sum()} duplicate rows that need to be removed.')
      file.drop_duplicates(inplace=True)
      file = file.drop_duplicates(subset=['film'],keep='first')
  except:
    raise RuntimeError(f'{colors.RED}A problem occured while deleting duplicates{colors.END}')
  print(f"{colors.GREEN}DUPLICATE ROWS HAVE BEEN SUCCESFULLY DELETED!{colors.END}")
  return file

def externalKnowledge(file):
  ia = IMDb()
  i=2
  for movieTitle in file['film']:
    movies = ia.search_movie(f"{movieTitle}")
    try:
      if movies:
          movie = ia.get_movie(movies[0].movieID)
          print(f"Title: {movie['title']}")
          year=movie['year']
          rating=movie['rating']
      else:
          print("Movie not found.")
          continue
      print(i," ",rating)
      if not rating:
        file.at[i-2, 'imdb rating'] = np.nan
        continue
      else:
        file.at[i-2, 'imdb rating'] = rating
        file.at[i-2,'year']=year
    except:
      file.at[i-2, 'imdb rating'] = np.nan
      continue
    i=i+1
  print(f'{colors.GREEN}EXTERNAL KNOWLEDGE \'IMDb\' HAS BEEN SUCCESFULLY ADDED{colors.END}')
  # file.to_excel("clone.xlsx")
  return file


def stringMissingValues(file):
  try:
    for item in STRING_COL:
      if item not in file.columns:
        # print("ïtem",item)
        STRING_COL.remove(item)
        continue
    for index, row in file.iterrows():
        # oscar winners
        if 'oscar winners' in file.columns:
          if (pd.isnull(row['oscar winners'])):
            file.loc[index, 'oscar winners'] = 0
          else:
            file.loc[index, 'oscar winners'] = 1
        try:
          if pd.isnull(row['genre']):
            ia=IMDb()
            movies = ia.search_movie(row['film'])
            if movies:
                movie = ia.get_movie(movies[0].movieID)
                genre=",".join(movie['genres']).lower()
            else:
                print("Movie not found.")
                genre=np.nan
            file.at[index, 'genre'] = genre
        except:
            raise RuntimeError(f'{colors.RED}A problem occured while receiving external knowledge in string missing values{colors.END}')
    file['genre']=file['genre'].ffill()
    file['genre'] = file['genre'].str.replace(',', ' ').str.replace('.', ' ').str.replace('\s+', ' ', regex=True).str.strip()
    file['script type']=file['script type'].ffill()
    file['script type'] = file['script type'].str.replace(',', ' ').str.replace('.', ' ').str.replace('\s+', ' ', regex=True).str.strip()
    file['release date (us)']=file['release date (us)'].ffill()
  except:
    raise RuntimeError(f'{colors.RED}A problem occured while processing string missing values{colors.END}')

  print(f"{colors.GREEN}OTHER MISSING VALUES HAS BEEN SUCCESFULLY RESTORED!{colors.END}")

  return file

def numericMissingValues(file): # replacing ',' and missing values with the mean of the year it belongs to
  for item in COMMA_COL:
    if item not in file.columns:
      COMMA_COL.remove(item)
      continue
  if len(COMMA_COL)==0:
    print(f'{colors.GREEN}NO NUMERIC COLUMNS TO BE PROCESSED.SUCCESFULLY COMPLETED{colors.END}')
    return file
  try:
    file[COMMA_COL] = file[COMMA_COL].replace(',', '', regex=True)
    file['year']=pd.to_numeric(file['year'],errors='coerce')
    for element in COMMA_COL:
        # file[element]=file[element].astype(float)
        file[element] = pd.to_numeric(file[element], errors='coerce')
        for index, row in file.iterrows():
          year = row['year']
          mean = file[file['year'] == year][element].mean()
          file[element] = pd.to_numeric(file[element], errors='coerce')
          if np.isnan(row[element]):
            if np.isnan(mean):
              file[element]=file[element].ffill()
              continue
            file.loc[index, element] = mean
        file[element].bfill(inplace=True)
  except:
    raise ValueError(f'{colors.RED}A problem occured while processing numeric missing values{colors.END}')
  print(f"{colors.GREEN}NUMERIC MISSING VALUES HAS BEEN SUCCESFULLY RESTORED!{colors.END}")
  return file

def scaling(file):
  for item in SCALING_COL:
    if item not in SCALING_COL:
      SCALING_COL.remove(item)
      continue
  if len(SCALING_COL)!=0:
    try:
      scaler = StandardScaler()
      file[SCALING_COL] = scaler.fit_transform(file[SCALING_COL])
    except:
      raise ValueError(f'{colors.RED}A problem occured while scaling values{colors.END}')
  print(f"{colors.GREEN}SCALING HAS BEEN SUCCESFULLY COMPLETED!{colors.END}")
  return file

def normalization(file):
  for item in SCALING_COL:
    if item not in SCALING_COL:
      SCALING_COL.remove(item)
      continue
  if len(SCALING_COL)!=0:
    try:
      columns_to_normalize = file[SCALING_COL]
      scaler = MinMaxScaler()
      normalized_columns = pd.DataFrame(scaler.fit_transform(columns_to_normalize), columns=SCALING_COL)
      file[SCALING_COL] = normalized_columns
    except:
      raise ValueError(f'{colors.RED}A problem occured while normalising values{colors.END}')
  print(f"{colors.GREEN}NORMALISING HAS BEEN SUCCESFULLY COMPLETED!{colors.END}")
  return file



# CLASS
class DataPreprocessor():

  def __init__(self, fileToProcess, cloneProcessedFile):
        self.fileToProcess = fileToProcess
        self.cloneProcessedFile = cloneProcessedFile

        # You can perform any initialization here

  # DATA PREPROCESSING
  def executePreprocess(self,type=None):
    df=pd.read_excel(self.fileToProcess, sheet_name = 'Sheet1',na_values=['-','0'])
    df.columns = df.columns.str.lower().str.replace(r'\s+', ' ', regex=True)
    df.columns = df.columns.str.strip()
    df.to_excel(self.fileToProcess, index=False)

    df=dropUseless(df,USELESS_COL) # DELETE USELESS COLUMNS
    # externalKnowledge(df) # RETRIEVE EXTERNAL KNOWLEDGE, ADDITIONAL DATA
    df=precentagesToDecimal(df)# CONVERT PRECENTAGES CELLS TO DECIMAL
    df=numericMissingValues(df) # RETRIEVE MISSING VALUES
    df=stringMissingValues(df) # RETRIEVE MISSING VALUES
    df=oneHotEncoding(df) # ONE HOT ENCODING
    df=deleteDuplicate(df)  # CHECK FOR DUPLICATE ROWS
    df=dropUseless(df,['film','year']) # DELETE USELESS COLUMNS
    if type=='normalisation':
      df=normalization(df)
    elif type=='scaling':
      df=scaling(df)
    # df=scaling(df)
    df.to_excel(self.cloneProcessedFile) # Convert pandas updated dataset to a new excel with the final data
    # missing_data = pd.read_excel(self.cloneProcessedFile).isnull().sum()
    # print(f"# of missing data: {missing_data}")
    # print(df.describe().T)
    print(f"------------------------PRE-PROCESSING-FINISHED-------------------------\n")
    return df

if __name__=='__main__':
  # dp=DataPreprocessor("./movies_test _anon_sample.xlsx","sample.xlsx")
  dp=DataPreprocessor("Book.xlsx","datesFour.xlsx")
  dataset=dp.executePreprocess()
  print(dataset.head())
  print(dataset.columns)
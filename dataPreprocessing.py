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


ALL_NUMERIC=['year', 'rotten tomatoes critics', 'metacritic critics', 'average critics', 'rotten tomatoes audience', 'metacritic audience', 'rotten tomatoes vs metacritic deviance', 'average audience', 'audience vs critics deviance', 'opening weekend', 'opening weekend ($million)', 'domestic gross', 'domestic gross ($million)','foreign gross ($million)', 'foreign gross', 'worldwide gross', 'worldwide gross ($million)', 'budget ($million)','of gross earned abroad', 'budget recovered','budget recovered opening weekend','imdb rating','distributor','imdb vs rt disparity']
NO_TRAGET_STRINGS=['script type','primary genre','genre','release date (us)'] #except 'film' 'oscar winners','oscar detail'
TARGET_STRINGS=['oscar winner','oscar detail']
TYPES=['adaptation','original','based on a true story','sequel','remake']
USELESS_COL=['rotten tomatoes critics','metacritic critics','rotten tomatoes audience','metacritic audience','rotten tomatoes vs metacritic deviance','audience vs critics deviance','primary genre','opening weekend ($million)','domestic gross ($million)','foreign gross ($million)','worldwide gross ($million)','worldwide gross','budget recovered opening weekend','distributor','imdb vs rt disparity']

class colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    END = '\033[0m'

def oneHotEncoding(file):
  # SCRIPT TYPE
  if 'script type' in file.columns:
    try:
      one_hot_encoded = pd.get_dummies(file['script type'].apply(lambda x: next((t for t in TYPES if str(x).startswith(t)), None))).astype(int)
      file = pd.concat([file, one_hot_encoded], axis=1)
      file=dropUseless(file,['script type'])
    except:
      raise RuntimeError(f'{colors.RED}A problem occured while one-hot encoding script-type{colors.END}')

  # OSCAR DETAILS
  if 'oscar detail' in file.columns:
    try:
      one_hot_encoded = file['oscar detail'].str.get_dummies(', ').astype(int)
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
      # print('Set:::',genres)
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

def externalIMDb(file):
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

def externalGenre(file):
  ia=IMDb()
  try:
    for index,row in file.iterrows():
      if pd.isna(row['genre']):
        movies = ia.search_movie(row['film'])
        if movies:
            movie = ia.get_movie(movies[0].movieID)
            genre=",".join(movie['genres']).lower()
        else:
            print("Movie not found.")
            genre=pd.isna
        file.at[index, 'genre'] = genre
  except:
      raise RuntimeError(f'{colors.RED}A problem occured while receiving external knowledge in string missing values{colors.END}')
  return file


def stringMissingValues(file):
  try:
    for item in NO_TRAGET_STRINGS:
      if item not in file.columns:
        NO_TRAGET_STRINGS.remove(item)
        print(f"{item} column removed from the array because it doesn't exist in the dataset")
        continue
      if len(ALL_NUMERIC)==0:
        print(f'{colors.GREEN}NO STRING COLUMNS TO BE PROCESSED.SUCCESFULLY COMPLETED{colors.END}')
        return file
    file=externalGenre(file)
    for index, row in file.iterrows():
        if 'oscar winners' in file.columns:
          if (pd.isna(row['oscar winners'])):
            file.loc[index, 'oscar winners'] = 0
          else:
            file.loc[index, 'oscar winners'] = 1
    for j in NO_TRAGET_STRINGS:
      if file[j].isnull().any():
        file[j]=file[j].ffill()
      else:
        file[j]=file[j].bfill()
  except:
    raise RuntimeError(f'{colors.RED}A problem occured while processing string missing values{colors.END}')
  print(f"{colors.GREEN}OTHER MISSING VALUES HAS BEEN SUCCESFULLY RESTORED!{colors.END}")
  return file

def numericMissingValues(file): # replacing ',' and missing values with the mean of the year it belongs to
  try:
    for item in ALL_NUMERIC:
      if item not in file.columns:
        ALL_NUMERIC.remove(item)
        continue
    if len(ALL_NUMERIC)==0:
      print(f'{colors.GREEN}NO NUMERIC COLUMNS TO BE PROCESSED.SUCCESFULLY COMPLETED{colors.END}')
      return file
  # file=externalIMDb(file) # IMDb
    for element in ALL_NUMERIC:
      for index, row in file.iterrows():
        year = row['year']
        mean = file[file['year'] == year][element].mean()
        file[element] = pd.to_numeric(file[element], errors='coerce')
        if pd.isna(row[element]):
          if pd.isna(mean):
            file[element]=file[element].ffill()
            continue
          file.loc[index, element] = mean
      file[element].bfill(inplace=True)
  except:
    print(element,index,row,mean)
    raise ValueError(f'{colors.RED}A problem occured while processing numeric missing values{colors.END}')
  print(f"{colors.GREEN}NUMERIC MISSING VALUES HAS BEEN SUCCESFULLY RESTORED!{colors.END}")
  return file


def scaling(file):
  for item in ALL_NUMERIC:
    if item not in ALL_NUMERIC:
      ALL_NUMERIC.remove(item)
      continue
  if len(ALL_NUMERIC)!=0:
    try:
      scaler = StandardScaler()
      file[ALL_NUMERIC] = scaler.fit_transform(file[ALL_NUMERIC])
    except:
      raise ValueError(f'{colors.RED}A problem occured while scaling values{colors.END}')
  print(f"{colors.GREEN}SCALING HAS BEEN SUCCESFULLY COMPLETED!{colors.END}")
  return file

def normalization(file):
  for item in ALL_NUMERIC:
    if item not in ALL_NUMERIC:
      ALL_NUMERIC.remove(item)
      continue
  if len(ALL_NUMERIC)!=0:
    try:
      columns_to_normalize = file[ALL_NUMERIC]
      scaler = MinMaxScaler()
      normalized_columns = pd.DataFrame(scaler.fit_transform(columns_to_normalize), columns=ALL_NUMERIC)
      file[ALL_NUMERIC] = normalized_columns
    except:
      raise ValueError(f'{colors.RED}A problem occured while normalising values{colors.END}')
  print(f"{colors.GREEN}NORMALISING HAS BEEN SUCCESFULLY COMPLETED!{colors.END}")
  return file

def columnDataFormating(file):
  try:
    #  Convert each value of the df that is ending with the '%' to decimal (divide by 100)
    file = file.map(lambda val: float(val.rstrip('%')) / 100 if isinstance(val, str) and val.endswith('%') else val)
  except:
    raise RuntimeError(f'{colors.RED}A problem occured while converting precentages to decimal.{colors.END}')
  try:
    file[ALL_NUMERIC] = file[ALL_NUMERIC].replace(',', '', regex=True).apply(pd.to_numeric, errors='coerce')
    file['budget ($million)'] = file['budget ($million)'] * 1000000
    # file[NO_TRAGET_STRINGS] = file[NO_TRAGET_STRINGS].replace(',', '', regex=True)
    file['genre'] = file['genre'].str.replace(',', ' ').str.replace('.', ' ').str.replace('\s+', ' ', regex=True).str.strip()
    # file['oscar detail'] = file['oscar detail'].str.extract(r'([^\(]+)')
    file['oscar detail'] = file['oscar detail'].str.split('(', n=1).str[0].str.strip()
  except:
    raise RuntimeError(f'{colors.RED}A problem occured while replacing charachters.{colors.END}')
  print(f"{colors.GREEN}ALL COLUMNS HAS BEEN SUCCESFULLY FORMATED!{colors.END}")
  return file

def initDataframe(xFile):
  try:
    df=pd.read_excel(xFile, sheet_name = 'Sheet1',na_values=['-','0'])
    df.columns = df.columns.str.lower().str.replace(r'\s+', ' ', regex=True)
    df = df.map(lambda x: x.lower() if isinstance(x, str) else x)
    df.columns = df.columns.str.strip()
    df.to_excel(xFile, index=False)
    for item in USELESS_COL:
      if item in ALL_NUMERIC:
        ALL_NUMERIC.remove(item)
      if item in NO_TRAGET_STRINGS:
        NO_TRAGET_STRINGS.remove(item)
  except:
    raise RuntimeError(f'{colors.RED}A problem occured while initializing the excel file!{colors.END}')
  print(f"{colors.GREEN}INITIALIZATION HAS BEEN SUCCESFULLY CONVERTED!{colors.END}")
  return df

def getCorrelation(items):
  correlation_matrix = items.corr()
  print(correlation_matrix)

# ------------------------------------------------------------------------------------------------------------------------
# CLASS
class DataPreprocessor():

  def __init__(self, fileToProcess, cloneProcessedFile):
        self.fileToProcess = fileToProcess
        self.cloneProcessedFile = cloneProcessedFile

  # DATA PREPROCESSING
  def executePreprocess(self,type=None):
    df=initDataframe(self.fileToProcess)
    # subset = df[['budget ($million)', 'budget recovered', 'budget recovered opening weekend']]
    # # subset = df[['rotten tomatoes critics',	'metacritic critics','average critics']]
    # # subset = df[['rotten tomatoes audience','metacritic audience','average audience']]
    # getCorrelation(subset)
    df=dropUseless(df,USELESS_COL) # DELETE USELESS COLUMNS
    df=columnDataFormating(df)
    df=numericMissingValues(df) # RETRIEVE MISSING VALUES
    df=stringMissingValues(df) # RETRIEVE MISSING VALUES
    df=columnDataFormating(df)
    df=oneHotEncoding(df) # ONE HOT ENCODING
    df=deleteDuplicate(df)  # CHECK FOR DUPLICATE ROWS
    df=dropUseless(df,['film','year']) # DELETE USELESS COLUMNS
    if type=='normalisation':
      df=normalization(df)
    elif type=='scaling':
      df=scaling(df)
    df.to_excel(self.cloneProcessedFile)
    missing_data = pd.read_excel(self.cloneProcessedFile).isnull().sum()
    # print(f"# of missing data: {missing_data}")
    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    #   print(f"# of missing data:\n{missing_data}")
    # print(df.describe().T)
    print(f"------------------------PRE-PROCESSING-FINISHED-------------------------\n")
    return df

if __name__=='__main__':
  # dp=DataPreprocessor("./movies_test _anon_sample.xlsx","sample.xlsx")
  dp=DataPreprocessor("Book.xlsx","datesFour.xlsx")
  dataset=dp.executePreprocess()
  if dataset.isna().any().any():
    print("DataFrame contains NaN values.")
  else:
      print("DataFrame does not contain NaN values.")
  print(dataset.head())
  # print(dataset.columns)
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


USELESS_COL=['rotten tomatoes critics','metacritic critics','rotten tomatoes audience','metacritic audience','rotten tomatoes vs metacritic deviance','audience vs critics deviance','primary genre','opening weekend ($million)','worldwide gross','worldwide gross ($million)','domestic gross ($million)','foreign gross ($million)','worldwide gross ($million)','of gross earned abroad','distributor','imdb vs rt disparity','oscar detail']
COMMA_COL=['average critics','average audience','opening weekend','foreign gross','domestic gross','budget ($million)','budget recovered','budget recovered opening weekend','imdb rating']
STRING_COL=['script type','genre','oscar winners']
# PRECENTAGES=['budget recovered','budget recovered opening weekend']
TYPES=['adaptation','screenplay','original','based on a true story','sequel','remake']

class colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    END = '\033[0m'

def oneHotEncoding(file):
  print(file.shape)
  # SCRIPT TYPE
  # print(TYPES)
  #  for type in TYPES:
  #     # TODO fix the split method
  #     file[type] = file['script type'].apply(lambda )
  segment_text = lambda text, types: \
    (lambda r_text, res: res if not r_text else r_text[1:].strip() and segment_text(r_text[1:].strip(), types) if not any(r_text.startswith(t) for t in sorted(types, key=len, reverse=True)) else segment_text(r_text[len([t for t in sorted(types, key=len, reverse=True) if r_text.startswith(t)][0]):].strip(), types))(text.strip(), [])
  for t in TYPES:
    file[t] = file['script type'].apply(lambda x: 1 if t in segment_text(x, TYPES) else 0)

  file=dropUseless(file,['script type'])


  # GENRE
  genres=set()

  for filmGenres in file['genre']:
      corrected_genres = []
      for genre in filmGenres.split():
          corrected_genre = str(Word(genre).correct().lower())
          corrected_genres.append(corrected_genre)
      # TODO FIX IT THE SPELLING THREE LETTERS SIMILARITY
      final_genres = []
      for genre in corrected_genres:
        if len(final_genres) == 0:
            final_genres.append(genre)
        else:
            last_genre = final_genres[-1]
            if len(set(genre) - set(last_genre)) <= 1:
                consecutive_count = 0
                for i in range(len(genre) - 2):
                    if genre[i] == genre[i + 1] == genre[i + 2]:
                        consecutive_count += 1
                if consecutive_count < 3:
                    final_genres.append(genre)
            else:
                final_genres.append(genre)

      genres.update(final_genres)
      # genres.update(corrected_genres.split())

  # print(genres)
  try:
    for genre in genres:
        file[genre] = file['genre'].apply(lambda x: 1 if genre in x.split() else 0)
    file=dropUseless(file,['genre'])
  except:
    raise RuntimeError(f'{colors.RED}A problem occured while one-hot encoding genres{colors.END}')

  # DATE
  try:
    file['release date (us)'] = pd.to_datetime(file['release date (us)'], format='mixed')
    dateColumns = pd.DataFrame({
      'Month': file['release date (us)'].dt.month.astype(str),
      'Day': file['release date (us)'].dt.day.astype(str),
      'Year': file['release date (us)'].dt.year.astype(str)
      })
    encoded_dates = pd.get_dummies(dateColumns).astype(int)
    file = pd.concat([file, encoded_dates], axis=1)
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

def NumericalToNominal(file):
  print(f"Numerical to Nominal")
  return file

def dropUseless(file,uselessColumns):
  try:
    if len(uselessColumns)!=0:
      file.drop(uselessColumns,axis=1,inplace=True) # deleting useless data from the excel
  except:
    raise RuntimeError(f'{colors.RED}A problem occured while receiving external knowledge in string missing values{colors.END}')
  print(f"{colors.GREEN}USELESS COLUMNS HAS BEEN SUCCESFULLY DELETED!{colors.END}")
  return file


def deleteDuplicate(file):
  try:
    if(file.duplicated().sum()!=0):
      print(f'The dataset contains {file.duplicated().sum()} duplicate rows that need to be removed.')
      file.drop_duplicates(inplace=True)
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
      else:
        file.at[i-2, 'imdb rating'] = rating
        file.at[i-2,'year']=year
    except:
      file.at[i-2, 'imdb rating'] = np.nan
      continue
    # print(file.head(5))
    i=i+1
  print(f'{colors.GREEN}EXTERNAL KNOWLEDGE \'IMDb\' HAS BEEN SUCCESFULLY ADDED{colors.END}')
  # file.to_excel("clone.xlsx")
  return file


def stringMissingValues(file):
  try:
    for index, row in file.iterrows():
        # oscar winners
        if (pd.isnull(row['oscar winners'])):
          file.loc[index, 'oscar winners'] = 0
        else:
          file.loc[index, 'oscar winners'] = 1
        # Genre
        # file['genre']=file['genre'].str.lower()
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
    file['script type']=file['script type'].ffill()
    file['genre'] = file['genre'].str.replace(',', ' ').str.replace('.', ' ').str.replace('\s+', ' ', regex=True).str.strip()
    file['script type'] = file['script type'].str.replace(',', ' ').str.replace('.', ' ').str.replace('\s+', ' ', regex=True).str.strip()
  except:
    raise RuntimeError(f'{colors.RED}A problem occured while processing string missing values{colors.END}')

  print(f"{colors.GREEN}OTHER MISSING VALUES HAS BEEN SUCCESFULLY RESTORED!{colors.END}")

  return file

def numericMissingValues(file): # replacing ',' and missing values with the mean of the year it belongs to
  if len(COMMA_COL)==0:
    print(f'{colors.GREEN}NO NUMERIC COLUMNS TO BE PROCESSED.SUCCESFULLY COMPLETED{colors.END}')
    return file
  try:
    file[COMMA_COL] = file[COMMA_COL].replace(',', '', regex=True)
    for element in COMMA_COL:
        for index, row in file.iterrows():
          year = row['year']
          mean = file[file['year'] == year][element].mean()
          file[element] = pd.to_numeric(file[element], errors='coerce')
          if pd.isna(row[element]):
            file.loc[index, element] = mean
  except:
    raise ValueError(f'{colors.RED}A problem occured while processing numeric missing values{colors.END}')
  print(f"{colors.GREEN}NUMERIC MISSING VALUES HAS BEEN SUCCESFULLY RESTORED!{colors.END}")
  return file

SCALING_COL=['average critics','average audience','opening weekend','domestic gross','foreign gross','budget ($million)','budget recovered','budget recovered opening weekend','imdb rating'
]
def scaling(file):
  # print(SCALING_COL)
  try:
    scaler = StandardScaler()
    file[SCALING_COL] = scaler.fit_transform(file[SCALING_COL])
  except:
    raise ValueError(f'{colors.RED}A problem occured while scaling values{colors.END}')
  print(f"{colors.GREEN}SCALING HAS BEEN SUCCESFULLY COMPLETED!{colors.END}")
  return file

def normalization(file):
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
    df=pd.read_excel(self.fileToProcess, sheet_name = 'Sheet1',na_values=['-'])
    # print(df.head())
    df.columns = df.columns.str.lower().str.replace(r'\s+', ' ', regex=True)
    # print(df.head())
    df.columns = df.columns.str.strip()
    df.to_excel(self.fileToProcess, index=False)

    df=dropUseless(df,USELESS_COL) # DELETE USELESS COLUMNS
    # externalKnowledge(df) # RETRIEVE EXTERNAL KNOWLEDGE, ADDITIONAL DATA
    df=precentagesToDecimal(df)# CONVERT PRECENTAGES CELLS TO DECIMAL
    df=numericMissingValues(df)
    df=stringMissingValues(df)
    df=oneHotEncoding(df)
    df=dropUseless(df,['film','year']) # DELETE USELESS COLUMNS
    if type=='normalisation':
      df=normalization(df)
    elif type=='scaling':
      df=scaling(df)
    df=deleteDuplicate(df)  # CHECK FOR DUPLICATE ROWS
    df.to_excel(self.cloneProcessedFile) # Convert pandas updated dataset to a new excel with the final data
    missing_data = pd.read_excel(self.cloneProcessedFile).isnull().sum()
    print(f"# of missing data: {missing_data}")
    print(f"--------------------------FINISHED-------------------------")
    return df

if __name__=='__main__':
  dp=DataPreprocessor("Book.xlsx","datesFour.xlsx")
  dataset=dp.executePreprocess()
  print(dataset.head())
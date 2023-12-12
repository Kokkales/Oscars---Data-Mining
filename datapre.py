import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MultiLabelBinarizer
from textblob import Word
from imdb import IMDb
from openpyxl import load_workbook, Workbook
class colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    END = '\033[0m'

def uniqueGenres(file):

  file['genre'] = file['genre'].apply(lambda x: ' '.join([Word(word).correct().lower() for word in x.split()]))
  file['genre'] = file['genre'].apply(lambda x: re.sub(r'[^a-zA-Z,\s]', '', x))
# Save the corrected file back to the Excel file
  file.to_excel('new.xlsx', index=False)
  print("difficult job done ")

  all_genres = []
  for genres in file['genre']:
      if isinstance(genres, str):  # Check if the value is a string
          genres_list = re.split(',| ', genres)  # Split by comma or space
          all_genres.extend([genre.strip() for genre in genres_list if genre.strip()])  # Remove empty strings and strip spaces

  # Get unique genres
  unique_genres = list(set(all_genres))
  correct_unique=[]
  for word in unique_genres:
    correctWord=Word(word).correct().lower()
    correct_unique.append(str(correctWord))

  # Display unique genres
  # print(unique_genres)
  # print(correct_unique)
  unique_genres=list(set(correct_unique))
  # print(unique_genres)

  unique_words = list(unique_genres)
  for i in range(len(unique_genres)):
    for j in range(i + 1, len(unique_genres)):
        match_count = 0
        for k in range(min(len(unique_genres[i]), len(unique_genres[j]))):
            if unique_genres[i][k] == unique_genres[j][k]:
                match_count += 1
                if match_count > 3:
                    # print(unique_genres[i],unique_genres[j])
                    if unique_genres[i] in unique_words:
                        unique_words.remove(unique_genres[i])
                    elif (unique_genres[j] in unique_words):
                        unique_words.remove(unique_genres[j])
                    break
            else:
                break

  print("Original list:", unique_genres)
  print("Updated list:", unique_words)

  return unique_words

TYPES=['adaptation','screenplay','original','based on a true story','sequel','remake']
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
  for genre in genres:
      file[genre] = file['genre'].apply(lambda x: 1 if genre in x.split() else 0)

  file=dropUseless(file,['genre'])

  # DATE
  file['release date (us)'] = pd.to_datetime(file['release date (us)'], format='mixed')
  dateColumns = pd.DataFrame({
    'Month': file['release date (us)'].dt.month.astype(str),
    'Day': file['release date (us)'].dt.day.astype(str),
    'Year': file['release date (us)'].dt.year.astype(str)
    })
  encoded_dates = pd.get_dummies(dateColumns).astype(int)
  file = pd.concat([file, encoded_dates], axis=1)
  file=dropUseless(file,['release date (us)'])
  #  print(file)
  print(file.shape)
  #  print(file['release date (us)'])
  return file

def precentagesToDecimal(file):
  PRECENTAGES=['budget recovered','budget recovered opening weekend']
  #  Convert each value of the df that is ending with the '%' to decimal (divide by 100)
  file = file.map(lambda val: float(val.rstrip('%')) / 100 if isinstance(val, str) and val.endswith('%') else val)
  print(f"{colors.GREEN}PRECENTAGES HAS BEEN SUCCESFULLY CONVERTED!{colors.END}")
  # print(file[' Budget recovered'])
  # file.to_excel("clone.xlsx")
  return file

def NumericalToNominal(file):
  print(f"Numerical to Nominal")
  return file

def dropUseless(file,uselessColumns):
  file.drop(uselessColumns,axis=1,inplace=True) # deleting useless data from the excel
  print(f"{colors.GREEN}USELESS COLUMNS HAS BEEN SUCCESFULLY DELETED!{colors.END}")
  return file


def deleteDuplicate(file):
  print(f'The dataset contains {file.duplicated().sum()} duplicate rows that need to be removed.')
  file.drop_duplicates(inplace=True)
  print(f"{colors.GREEN}DUPLICATE ROWS HAVE BEEN SUCCESFULLY DELETED!{colors.END}")
  return file

def externalKnowledge(file):
  ia = IMDb()
  # test=print(df['Film'])
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
  STRING_COL=['script type','genre','oscar winners']
  for index, row in file.iterrows():
      # oscar winners
      if (pd.isnull(row['oscar winners'])):
        file.loc[index, 'oscar winners'] = 0
      else:
        file.loc[index, 'oscar winners'] = 1
      # Get the year of the film
      year = row['year']

      # Genre
      # file['genre']=file['genre'].str.lower()
      if pd.isnull(row['genre']):
        ia=IMDb()
        try:
          # print("::",row['film'])
          movies = ia.search_movie(row['film'])
          if movies: #TODO fix it
              movie = ia.get_movie(movies[0].movieID)
              genre=",".join(movie['genres']).lower()
              if not genre:
                  file.at[index, 'genre'] = np.nan
              else:
                  file.at[index, 'genre'] = genre
          else:
              print("Movie not found.")
              genre=np.nan
        except:
            file['genre']=file['genre'].fillna(method='ffill')
            # file.at[index, 'genre'] = np.nan
      if pd.isnull(row['script type']):
        file['script type']=file['script type'].fillna(method='ffill')
  file['genre'] = file['genre'].str.replace(',', ' ').str.replace('.', ' ').str.replace('\s+', ' ', regex=True).str.strip()
  file['script type'] = file['script type'].str.replace(',', ' ').str.replace('.', ' ').str.replace('\s+', ' ', regex=True).str.strip()

  print(f"{colors.GREEN}OTHER MISSING VALUES HAS BEEN SUCCESFULLY RESTORED!{colors.END}")

  return file

def numericalMissingValues(file):
  COMMA_COL=['average critics','average audience','opening weekend','foreign gross','worldwide gross','budget ($million)','budget recovered','budget recovered opening weekend','imdb rating']
  file[COMMA_COL] = file[COMMA_COL].replace(',', '', regex=True)
  for element in COMMA_COL:
      for index, row in file.iterrows():
        year = row['year']
        mean = file[file['year'] == year][element].mean()
        file[element] = pd.to_numeric(file[element], errors='coerce')
        if pd.isna(row[element]):
          file.loc[index, element] = mean
  print(f"{colors.GREEN}NUMERIC MISSING VALUES HAS BEEN SUCCESFULLY RESTORED!{colors.END}")
  return file
class DataPreprocessor():

  def __init__(self, fileToProcess, cloneProcessedFile):
        self.fileToProcess = fileToProcess
        self.cloneProcessedFile = cloneProcessedFile
        # You can perform any initialization here

  # DATA PREPROCESSING
  def executePreprocess(self):
    df=pd.read_excel(self.fileToProcess, sheet_name = 'Sheet1',na_values=['-'])
    # print(df.head())
    df.columns = df.columns.str.lower().str.replace(r'\s+', ' ', regex=True)
    df.columns = df.columns.str.strip()
    df.to_excel(self.fileToProcess, index=False)
    # print(df.shape)

    # 3. POINT USELESS COLUMNS
    USELESS_COL=['rotten tomatoes critics','metacritic critics','rotten tomatoes audience','metacritic audience','rotten tomatoes vs metacritic deviance','audience vs critics deviance','primary genre','opening weekend ($million)','domestic gross','domestic gross ($million)','foreign gross ($million)','worldwide gross ($million)','of gross earned abroad','distributor','imdb vs rt disparity','oscar detail']
    df=dropUseless(df,USELESS_COL) # 4. DELETE USELESS COLUMNS
    # externalKnowledge(df) # 5. RETRIEVE EXTERNAL KNOWLEDGE, ADDITIONAL DATA
    df=precentagesToDecimal(df)# 6. CONVERT PRECENTAGES CELLS TO DECIMAL
    df=numericalMissingValues(df)
    df=stringMissingValues(df)
    df=oneHotEncoding(df)

    # df=deleteDuplicate(df)  # 9. CHECK FOR DUPLICATE ROWS
    df=dropUseless(df,['film','year']) # 4. DELETE USELESS COLUMNS
    # print(df.head(5))
    print(df.shape)
    df.to_excel(self.cloneProcessedFile) #10. Convert pandas updated dataset to a new excel with the final data
    missing_data = pd.read_excel(self.cloneProcessedFile).isnull().sum()
    # print(f"# of missing data: {missing_data}")
    print(f"--------------------------FINISHED-------------------------")
    return df
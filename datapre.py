import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from imdb import IMDb
from openpyxl import load_workbook, Workbook

# def columnFormat(file):

#   # Convert all values in each column to lowercase
#   file = file.apply(lambda col: col.astype(str).str.lower())
#   # Save the modified DataFrame back to Excel
#   file.to_excel('modified.xlsx', index=False)
#   print(f"Format file: {file}")
#   return file

def precentagesToDecimal(file):
  #  Convert each value of the df that is ending with the '%' to decimal (divide by 100)
  file = file.map(lambda val: float(val.rstrip('%')) / 100 if isinstance(val, str) and val.endswith('%') else val)
  print(f"PRECENTAGES HAS BEEN SUCCESFULLY CONVERTED!")
  # print(file[' Budget recovered'])
  # file.to_excel("clone.xlsx")
  return file

def NumericalToNominal(file):
   print(f"Numerical to Nominal")
   return file

def dropUseless(file,uselessColumns):
   file.drop(uselessColumns,axis=1,inplace=True) # deleting useless data from the excel
   print("USELESS COLUMNS HAS BEEN SUCCESFULLY DELETED!")
   return file


def deleteDuplicate(file):
  print(f'The dataset contains {df.duplicated().sum()} duplicate rows that need to be removed.')
  df.drop_duplicates(inplace=True)
  print(f"DUPLICATE ROWS HAVE BEEN SUCCESFULLY DELETED!")
  return file

def externalKnowledge(file):
  ia = IMDb()
  # test=print(df['Film'])
  i=2
  for movieTitle in file['Film']:
    movies = ia.search_movie(f"{movieTitle}")
    # movies = ia.search_movie(f"")
    # Get details of the first search result
    try:
      if movies:
          movie = ia.get_movie(movies[0].movieID)

          # Print movie details
          # print("Movie Details:")
          print(f"Title: {movie['title']}")
          # print(f"Year: {movie['year']}")
          year=movie['year']
          # print(f"Rating: {movie['rating']}")
          rating=movie['rating']
          # print(f"Genres: {', '.join(movie['genres'])}")
          # print(f"Plot: {movie['plot'][0]}")
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
    # file.to_excel("clone.xlsx")
    print(file.head(5))
    i=i+1
  print(f'EXTERNAL KNOWLEDGE \'IMDb\' HAS BEEN SUCCESFULLY ADDED')
  return file

def missingValues(file):
  COMMA_COL=['average critics ','average audience ','opening weekend','domestic gross','foreign gross','worldwide gross',' of gross earned abroad','budget ($million)',' budget recovered',' budget recovered opening weekend']
  file[COMMA_COL] = file[COMMA_COL].replace(',', '', regex=True)
  for index, row in file.iterrows():
      # Get the year of the film
      year = row['year']

      # Calculate mean for films created in the same year
      # mean of average critics
      mean_critics = file[file['year'] == year]['average critics '].mean()
      file['average critics '] = pd.to_numeric(df['average critics '], errors='coerce')
      mean_audience = file[file['year'] == year]['average audience '].mean()
      file['average audience '] = pd.to_numeric(df['average audience '], errors='coerce')
      mean_opening = file[file['year'] == year]['opening weekend'].mean()
      file['opening weekend'] = pd.to_numeric(df['opening weekend'], errors='coerce')
      mean_domestic = file[file['year'] == year]['domestic gross'].mean()
      file['domestic gross'] = pd.to_numeric(df['domestic gross'], errors='coerce')
      mean_foreign = file[file['year'] == year]['foreign gross'].mean()
      file['foreign gross'] = pd.to_numeric(df['foreign gross'], errors='coerce')
      mean_worldwide = file[file['year'] == year]['worldwide gross'].mean()
      file['worldwide gross'] = pd.to_numeric(df['worldwide gross'], errors='coerce')
      mean_abroad = file[file['year'] == year][' of gross earned abroad'].mean()
      file[' of gross earned abroad'] = pd.to_numeric(df[' of gross earned abroad'], errors='coerce')
      mean_budget = file[file['year'] == year]['budget ($million)'].mean()
      file['budget ($million)'] = pd.to_numeric(df['budget ($million)'], errors='coerce')
      mean_budget_rec = file[file['year'] == year][' budget recovered'].mean()
      file[' budget recovered'] = pd.to_numeric(df[' budget recovered'], errors='coerce')
      mean_budget_rec_open = file[file['year'] == year][' budget recovered opening weekend'].mean()
      file[' budget recovered opening weekend'] = pd.to_numeric(df[' budget recovered opening weekend'], errors='coerce')

      # Fill missing value with the mean for that year
      if pd.isna(row['average critics ']):
          file.loc[index, 'average critics '] = mean_critics
      if pd.isna(row['average audience ']):
          file.loc[index, 'average audience '] = mean_audience
      if pd.isna(row['opening weekend']):
          file.loc[index, 'opening weekend'] = mean_opening
      if pd.isna(row['domestic gross']):
          file.loc[index, 'domestic gross'] = mean_domestic
      if pd.isna(row['foreign gross']):
          file.loc[index, 'foreign Gross'] = mean_foreign
      if pd.isna(row['worldwide gross']):
          file.loc[index, 'worldwide gross'] = mean_worldwide
      if pd.isna(row[' of gross earned abroad']):
          file.loc[index, ' of gross earned abroad'] = mean_abroad
      if pd.isna(row['budget ($million)']):
          file.loc[index, 'budget ($million)'] = mean_budget
      if pd.isna(row[' budget recovered']):
          file.loc[index, ' budget recovered'] = mean_budget_rec
      if pd.isna(row[' budget recovered opening weekend']):
          file.loc[index, ' budget recovered opening weekend'] = mean_budget_rec_open
      # print("HPPP",index)
      # file.to_excel('test.xlsx')
      # print(index)
  print(mean_opening)
  return file

# DATA PREPROCESSING
# 1. Open Main excel using pandas
df = pd.read_excel('movies.xlsx')
# df.columns = df.columns.str.lower()
df.columns = df.columns.str.lower().str.replace(r'\s+', ' ', regex=True)

# Save the modified DataFrame back to Excel
df.to_excel('testing.xlsx', index=False)
# df.iloc[0] = df.iloc[0].astype(str).str.lower()
# Save the modified DataFrame back to Excel
# df.to_excel('testing.xlsx', index=False)

df=pd.read_excel('testing.xlsx', sheet_name = 'Sheet1',na_values=['-'])
print(df.shape)

#TODO 2. Format Column names
# df=columnFormat(df)
# 3. POINT USELESS COLUMNS
USELESS_COL=['rotten tomatoes critics','metacritic critics','rotten tomatoes audience ','metacritic audience ','rotten tomatoes vs metacritic deviance','audience vs critics deviance ','primary genre','opening weekend ($million)','domestic gross ($million)','foreign gross ($million)','worldwide gross ($million)','distributor','imdb vs rt disparity','oscar detail']
# print(len(USELESS_COL))
df=dropUseless(df,USELESS_COL) # 4. DELETE USELESS COLUMNS
print(f"New shape: {df.shape}")
# externalKnowledge(df) # 5. RETRIEVE EXTERNAL KNOWLEDGE, ADDITIONAL DATA
df=precentagesToDecimal(df)# 6. CONVERT PRECENTAGES CELLS TO DECIMAL
# TODO7. REPLACE MISSING VALUES WITH THE COLUMN MEAN
df=missingValues(df)
# TODO8. CONVERT NUMERICAL TO NOMINAL VALUES

df=deleteDuplicate(df)  # 9. CHECK FOR DUPLICATE ROWS
df=dropUseless(df,['film','year']) # 4. DELETE USELESS COLUMNS
print(df.shape)

df.to_excel("clone.xlsx") #10. Convert pandas updated dataset to a new excel with the final data
# print(df.columns)
missing_data = pd.read_excel('clone.xlsx').isnull().sum()
print(f"# of missing data: {missing_data}")
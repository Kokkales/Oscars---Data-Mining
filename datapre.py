import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from imdb import IMDb
from openpyxl import load_workbook, Workbook

# CONSTANTS
USELESS_COL=['Year','Rotten Tomatoes  critics','Metacritic  critics','Rotten Tomatoes Audience ','Metacritic Audience ','Rotten Tomatoes vs Metacritic  deviance','Audience vs Critics deviance ','Primary Genre','Opening weekend ($million)','Domestic gross ($million)','Foreign Gross ($million)','Worldwide Gross ($million)','Distributor','IMDB vs RT disparity','Oscar Detail']
# print(len(USELESS_COL))




# DATA PREPROCESSING
df=pd.read_excel('movies.xlsx')
print(df.shape)
df.drop(USELESS_COL,axis=1,inplace=True) # deleting useless data from the excel
print(df.shape)
# clone the pandas to excel
df.to_excel("clone.xlsx")
# print(df.columns)
# Create an instance of IMDb
# ia = IMDb()

# # test=print(df['Film'])
# i=2
# for movieTitle in df['Film']:
#   movies = ia.search_movie(f"{movieTitle}")
#   # movies = ia.search_movie(f"")
#   # Get details of the first search result
#   try:
#     if movies:
#         movie = ia.get_movie(movies[0].movieID)

#         # Print movie details
#         # print("Movie Details:")
#         print(f"Title: {movie['title']}")
#         # print(f"Year: {movie['year']}")
#         rating=movie['rating']
#         # print(f"Rating: {movie['rating']}")
#         # print(f"Genres: {', '.join(movie['genres'])}")
#         # print(f"Plot: {movie['plot'][0]}")
#     else:
#         print("Movie not found.")
#         continue
#     print(i," ",rating)
#     if not rating:
#       df.at[i-2, 'IMDb Rating'] = np.nan
#     else:
#       df.at[i-2, 'IMDb Rating'] = rating
#   except:
#      df.at[i-2, 'IMDb Rating'] = np.nan
#      continue
#   print(df.head(5))
#   i=i+1


# # check for duplicates
# print(f'The dataset contains {df.duplicated().sum()} duplicate rows that need to be removed.')
# df.drop_duplicates(inplace=True)



# print(df.head(40))
# print(len(df.columns))
# Delete completely empty columns
# for columnName in df.columns:
#   isEmpty=df[columnName].isnull().all()
#   if isEmpty:
#         print(f"The column '{columnName}' is empty.")
#         df.drop(f'{columnName}',axis=1,inplace=True)

# print(len(df.columns),f' empty columns deleted')

# Checking for duplicated rows
# print(df.head())
# print(df.shape)
# print(df.info())
# print(df.describe(include='object'))
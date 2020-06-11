#!/usr/bin/env python
# coding: utf-8

# # Importing the packages

# In[1]:


import operator
import pandas as pd
import seaborn as sns
import sqlite3
import matplotlib.pyplot as plt

from collections import Counter
from tqdm import tqdm

tqdm.pandas()
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Reading all the datasets.

# In[3]:


# Reading the actors dataset.
df_actors = pd.read_csv('dataset/Movie_Actors.csv')

# Reading the rating dataset.
df_ratings = pd.read_csv('dataset/Movie_AdditionalRating.csv')

# Reading the genres dataset.
df_genres = pd.read_csv('dataset/Movie_Genres.csv')

# Reading the movies dataset.
df_movies = pd.read_csv('dataset/Movie_Movies.csv')

# Reading the writers dataset.
df_writers = pd.read_csv('dataset/Movie_Writer.csv')


# In[ ]:





# # 1) Insights from the datasets.

# In[ ]:





# ## i) Insights about the Actors Dataset.

# In[4]:


# first 5 rows from the actors dataset to see the features of this dataset.
df_actors.head()


# In[5]:


print('Total Numbers of rows.', len(df_actors), '\n')
print('Information about the dataset.', '\n')
df_actors.info()


# In[6]:


#finding the actor who appears most frequently in movies.
actors_with_most_movies = dict(sorted(dict(Counter(df_actors.Actors.tolist())).items(), key=operator.itemgetter(1),reverse=True))


# In[7]:


# printing the first 10 actors with most number of movies.
for i, (name, times) in enumerate(actors_with_most_movies.items()):
    if i > 10:
        break
    print('The actor {} occured in {} movies.'.format(name, times))


# In[ ]:





# ## ii) insights about the Additional Ratings.

# In[8]:


# first 5 rows from the ratings dataset to see the features of this dataset.
df_ratings.head()


# In[9]:


print('Total Numbers of rows.', len(df_ratings), '\n')
print('Information about the dataset.', '\n')
df_ratings.info()


# In[10]:


# Total Number of Rating Sources.
print('Total Sources.', len(df_ratings.RatingSource.unique().tolist()), '\n')
print('Rating Sources are. ')
df_ratings.RatingSource.unique().tolist()


# In[11]:


# Frequency of iterations from each sources.
rating_Sources_counts = dict( sorted(dict(Counter(df_ratings.RatingSource.tolist())).items(), key=operator.itemgetter(1),reverse=True))
rating_Sources_counts


# In[12]:


# printing 5 rows of each rating sources.

rating_sources = df_ratings.RatingSource.unique().tolist()
for source in rating_sources:
    print('\n --------- Source {source}--------------- \n'.format(source=source))
    print(df_ratings[df_ratings.RatingSource == source][:2])


# In[ ]:





# ## iii) Insights about Genres Dataset.

# In[13]:


# first 5 rows from the genres dataset to see the features of this dataset.
df_genres.head()


# In[14]:


print('Total Numbers of rows.', len(df_genres), '\n')
print('Information about the dataset.', '\n')
df_genres.info()


# In[15]:


# Summary about the Genre column.
df_genres.Genre.describe()


# In[ ]:





# ## iv) Insights about Writers Dataset.

# In[ ]:





# In[16]:


# first 5 rows from the writers dataset to see the features of this dataset.
df_writers.head()


# In[17]:


print('Total Numbers of rows.', len(df_writers), '\n')
print('Information about the dataset.', '\n')
df_writers.info()


# In[18]:


# Writer with his/her number of responsibilties in this dataset.
writer_With_responsibilties = dict( sorted(dict(Counter(df_writers.Person.tolist())).items(), key=operator.itemgetter(1),reverse=True))


# In[19]:


# First 5 writers with their number of responsibilties.
for i, (writer, responsibilties) in enumerate(writer_With_responsibilties.items()):
    if i > 5:
        break
    print('The writer {writer} have {responsibilties} responsibilties'.format(writer=writer, responsibilties=responsibilties))


# In[20]:


# Types of Responsibilties in the dataset.
print('Total Responsibilties.', len(df_writers.Responsibility.unique().tolist()), '\n')
print('First 10 Responsibilties are. ')
df_writers.Responsibility.unique().tolist()[:10]


# In[ ]:





# ## v) Insights about the Movies.

# In[21]:



# first 5 rows from the Movies dataset to see the features of this dataset.
df_movies.head()


# In[22]:


print('Total Numbers of rows.', len(df_movies), '\n')
print('Information about the dataset.', '\n')
df_movies.info()


# In[23]:


# shape of the dataset, which represents the set of number of rows and columns
df_movies.shape


# In[25]:


# Checking the percentage of the non-null values in the dataset.
# it will give us the column name and the percentage of non-null values in that particular column.

# to perform analytics we remove those features from the dataset, which have less then 70% of non-null values. 

df_movies.notna().mean().round(4)*100


# In[26]:


# filtering those features which have less than 70% non-null values.
df_movies = df_movies[['Country', 'Director', 'Language', 'Plot', 'Released', 'Runtime',
                       'Title', 'Type', 'Year', 'imdbRating', 'imdbVotes', 'imdbID']]


# In[27]:


# Checking the shape of movie dataset, after removing the features having a big number of null values.
df_movies.shape


# In[28]:


# Providing the Summary of Country Features.
df_movies.Country.describe()


# In[29]:


# Providing the Summary of Director Features.
df_movies.Director.describe()


# In[30]:


# Providing the Summary of Language Features.
df_movies.Language.describe()


# In[31]:


# Providing the Summary of Released Features.
df_movies.Released.describe()


# In[32]:


# Providing the Summary of Year Features.
df_movies.Year.describe()


# In[33]:


# Providing the Summary of Rating Features.
df_movies.imdbRating.describe()


# In[ ]:





# # 2) Data Preprocessing

# In[ ]:





# ## i) Processing the Year Feature.

# In[ ]:





# In[34]:


# Formatting the Year Feature from the dataset.
df_movies['Year'] = df_movies[df_movies.Year.notna()][:].Year.progress_apply(lambda year: int(year) if type(year) == int else int(year) if type(year) == float else int(year.split('–')[0]))


# In[35]:


# Changing the Data type of Year Feature
df_movies = df_movies[df_movies.Year.notna()].astype({'Year': 'int32'})


# In[36]:


# Printing all the data types of movies dataset.
df_movies.dtypes


# In[ ]:





# ## ii) As we can see that there are so many movies, which have no rating at all, we can add more ratings by merging additional ratings from rating dateset using imdbID.
# 
# ### * but before merging, we have to use same scale for the each rating, because rating dataset don't have same rating process.
# 

# In[37]:


# printing 2 rows of each rating sources.

rating_sources = df_ratings.RatingSource.unique().tolist()
for source in rating_sources:
    print('\n --------- Source {source}--------------- \n'.format(source=source))
    print(df_ratings[df_ratings.RatingSource == source][:2])


# In[38]:


# This function will change the scale of ratings from different sources, into a one type of ratings.
def change_scale(row):
    try:
        # get ratings and strip for spaces.
        rating = row.Rating
        rating = rating.strip()
        # get sources
        sources = row.RatingSource
        # condition for those sources who uses percentage for rating.
        if sources ==' Rotten Tomatoes' or sources == 'Rotten Tomatoes':
            rating = float(rating.strip('%'))/10.0
        # condition for those sources who uses scale of 100 for rating.
        else:
            rating = float(rating.split('/')[0])
            if sources == ' Metacritic' or sources == 'Metacritic':
                rating = rating/10.0
        return rating
    except:
        # for debugging if any error occurs.
        import ipdb; ipdb.set_trace()


# In[39]:


# creating a new column into the rating dataset, for new rating. 
df_ratings['New_Ratings'] = df_ratings[df_ratings.Rating.notna()].progress_apply(lambda row: change_scale(row), axis=1)


# In[40]:


# printing 2 rows of each rating sources after rescaling.

rating_sources = df_ratings.RatingSource.unique().tolist()
for source in rating_sources:
    print('\n --------- Source {source}--------------- \n'.format(source=source))
    print(df_ratings[df_ratings.RatingSource == source][:2])


# In[41]:


# removing the old scaling and unnecessary column of unnamed: 0.
df_ratings = df_ratings[['New_Ratings', 'RatingSource', 'imdbID']]


# In[42]:


# Printing the First 2
df_ratings.head(2)


# In[43]:


# creating the dataset for ratings with null values.
df_empty_rating_movies = df_movies[df_movies.imdbRating.isna()]
# number of rows in the dataset which contains empty rating.
len(df_empty_rating_movies)


# In[46]:


# merging the ratings dataset with empty_rating_movies on behalf of imdbID.
df_merge_movie_ratings = pd.merge(df_empty_rating_movies, df_ratings, on='imdbID')
#checking the shape of the dataset
df_merge_movie_ratings.shape


# In[47]:


df_merge_movie_ratings.columns


# In[48]:


# adding these new created 
def adding_ratings(row):
    df_movies.loc[df_movies.imdbID == row.imdbID, 'imdbRating'] = row.New_Ratings


# In[49]:


df_merge_movie_ratings.progress_apply(lambda row: adding_ratings(row), axis=1)


# In[50]:


len(df_movies[df_movies.imdbRating.notna()])


# In[51]:


# sorting the movies dataset in descending order accoding to the ratings and display only the top 10.
df_movies.sort_values(by=['imdbRating'], ascending=False)[:10]


# In[ ]:





# ## iii) Merging the Actors dataset into the movies dataset using imdbID

# In[81]:


# reading the column names of the actors dataset.
df_actors.columns


# In[82]:


# removing the extra spaces from the begining and starting of the actors name.
df_actors['Actors'] = df_actors[df_actors.Actors.notna()].Actors.progress_apply(lambda actor: actor.strip())


# In[83]:


# After deleting the duplicates actors name.
print(len(df_actors))


# In[84]:


# merging the movies data set and actors datatset to create the df_movie_analysis dataset.
df_movie_analysis = pd.merge(df_movies, df_actors, on='imdbID', left_index=False)


# In[85]:


# print the percentage of non-empty dataset in the analysis dataset.
df_movie_analysis.notna().mean().round(4)*100


# ## iv) merging the Genres dataset into the movies dataset using imdbID.

# In[ ]:





# In[86]:


df_genres.columns


# In[88]:


# removing the extra spaces from the genres.
df_genres['Genre'] = df_genres[df_genres.Genre.notna()].Genre.progress_apply(lambda genre: genre.strip())


# In[89]:


# adding genres into the movies Data set by merging genre data on the behalf of the imdbID
df_movie_analysis = pd.merge(df_movie_analysis, df_genres, on='imdbID', left_index=False)


# In[90]:


df_movie_analysis.columns


# In[91]:


df_movie_analysis = df_movie_analysis[['Country', 'Director', 'Language', 'Plot', 'Released', 'Runtime',
       'Title', 'Type', 'Year', 'imdbRating', 'imdbVotes', 'imdbID', 'Actors', 'Genre']]


# In[92]:


len(df_movie_analysis)


# In[93]:


df_movie_analysis.head()


# In[94]:


df_movie_analysis.Genre.nunique()


# In[95]:


# print the percentage of non-empty dataset in the analysis dataset.
df_movie_analysis.notna().mean().round(4)*100


# ## v) Merging the writers dataset into the movies dataset using imdbID 

# In[96]:


# Reading the first two rows of the dataset to show the features of this dataset.
df_writers.head(2)


# In[97]:


# removing the extra spaces from the Person Feature of the writers dataset.
df_writers['Person'] = df_writers[df_writers.Person.notna()].Person.progress_apply(lambda person: person.strip())


# In[98]:


# printing the unique writers in the writers dataset.
df_writers.Person.nunique()


# In[99]:


# merging the movie_analysis dataset and the writers dataset to add new data into the movie analysis dataset.
df_movie_analysis = pd.merge(df_movie_analysis, df_writers, on='imdbID', how='inner', left_index=False)


# In[100]:


# filtering the features to be used in the analysis dataset.
df_movie_analysis = df_movie_analysis[['Country', 'Director', 'Language', 'Plot', 'Released', 'Runtime',
       'Title', 'Type', 'Year', 'imdbRating', 'imdbVotes', 'imdbID', 'Actors', 'Genre', 'Person', 'Responsibility']]


# In[102]:


# printing the percentage completeness of each feature of movies dataset before processing.
df_movies.notna().mean().round(4)*100


# In[103]:


# printing the percentage completeness of each feature of movies dataset after processing.
df_movie_analysis.notna().mean().round(4)*100


# In[104]:


# sample size of the movies_analysis dataset.
len(df_movie_analysis)


# In[ ]:





# # Analysis

# ## In this analysis portion we can run multiple analysis. e.g.
# 
# ### 1. Analysis On Directors.
# 
# #### i. Maximum number of movies directed by director(Top 10).
# 
# #### ii. Maximum number of movies directed by a director in a genre(Top 10).
# 
# #### iii. The year in which the director had directed the most movies.
# 
# 
# ### 2. Analysis On Actors.
# 
# #### i. Maximum number of movies performed by an Actor(Top 10).
# 
# #### ii. Maximum number of movies performent by an Actor in a genre(Top 10).
# 
# ### 3. Analysis On Writer.
# 
# #### i. Maximum number of movies written by writer(Top 10).
# 
# #### ii. Maximum number of movies written by writer in a genre(Top 10).
# 
# ### 4. Aanalysis On Rating.
# 
# #### i.  Most rating Genre.
# 
# ### 5. Genre.
# 
# #### i.  Maximum Number of movies in Genre.
# #### ii.  Maximum Number of movies in an year in a certain Genre.
# #### iii.  Maximum Number of movies of a language in Genre.

# In[ ]:





# ### 1) Analysis on Director

# In[135]:


# Checking the Top 10 Director with most number of movies
directors_with_movies_counts = dict( sorted(dict(Counter(df_movie_analysis.Director.tolist())).items(), key=operator.itemgetter(1),reverse=True))
for i, (director, num_movies) in enumerate(directors_with_movies_counts.items()):
    if i > 11: break
    if i > 0: print('{director} had directed {movies} movies.'.format(director=director, movies=num_movies))


# In[107]:


# Checking the Directors who directed most number of movies in a specific genre sorted by Genre Name.

movies_director_group = df_movie_analysis.groupby(['Director', 'Genre'])
top_director = movies_director_group['imdbID'].count().reset_index()
top_director.columns = ['Director', 'Genre', 'Number_of_Movies']
top_director.sort_values(by=['Number_of_Movies'], ascending=False, inplace=True)
top_director[:10].sort_values(by=['Genre'], ascending=True).head(10)


# In[108]:


# Creating a list of top 10 Directors.

top_directors = []
for i, (director, counts) in enumerate(directors_with_movies_counts.items()):
    if i > 11: break
    top_directors.append(director)
top_directors = top_directors[1:]
print(top_directors)


# In[109]:


# Creating a column for keep the status of top 10 directors.

df_movie_analysis['Keep_Director'] = df_movie_analysis[df_movie_analysis.Director.notna()].Director.progress_apply(lambda director: True if director in top_directors else False)


# In[110]:


# create a new dataframe of top 10 directors.

df_directors = df_movie_analysis[df_movie_analysis.Keep_Director == True]


# In[111]:


# length of the directors.
len(df_directors)


# In[112]:


# Filtering the director from the year 1980.

df_directors = df_directors[(df_directors.Year >= 1980)]


# In[131]:


# creating the group by filter from the feature of Year and Director.
movies_director_with_year_group = df_directors.groupby(['Year', 'Director']) 


# In[132]:


# showing the mean of the rating of the director of showing the average rating of the movie directed by a director in the year.

movies_director_with_year_group['imdbRating'].mean().reset_index()


# In[133]:


# showing the number of movies directed by the Director in one specific year.
movies_director_with_year_group['imdbID'].count().reset_index()#.sort_values(by=['imdbID'], ascending=False)


# In[ ]:





# ### 2) Movie Analysis on Genres.

# In[ ]:





# In[144]:


# creating the Dataframe containing the Genres and number of movies in that Genres.

movies_genres_group = df_movie_analysis.groupby('Genre')
top_genres = movies_genres_group['imdbID'].count().reset_index()
top_genres.columns = ['Genre', 'Number_of_Movies']
top_genres.sort_values(by=['Number_of_Movies'], ascending=False, inplace=True)
top_genres.head(5)


# In[154]:


# creating the list of top 5 genres.
top_five_genres = top_genres.Genre.unique().tolist()[:5]


# In[151]:


# Creating a column for keep the status of top 5 Genres.
df_movie_analysis['Keep_Genre'] = df_movie_analysis[df_movie_analysis.Genre.notna()].Genre.progress_apply(lambda genre: True if genre in top_five_genres else False)


# In[153]:


# creating a dataset from top 5 genres.
df_genres_analysis = df_movie_analysis[df_movie_analysis.Keep_Genre == True]


# In[161]:


# Filtering the dataset according to the date.
df_genres_analysis = df_genres_analysis[df_genres_analysis.Year >= 2000]
movies_genres_with_year_group = df_genres_analysis.groupby(['Year', 'Genre'])


# In[162]:


# Movies plot according to the Genre and their rating according to a specific year.

fig, ax = plt.subplots()

fig.set_size_inches(21, 11)
sns.pointplot(x = 'Year', y = 'imdbRating', data = movies_genres_with_year_group['imdbRating'].mean().reset_index(), hue = 'Genre', ax = ax)


# In[163]:


# Movies plot according to the Genre and number of movies genereated in a specific year.

fig, ax = plt.subplots()

fig.set_size_inches(21, 11)
sns.pointplot(x = 'Year', y = 'imdbID', data = movies_genres_with_year_group['imdbID'].count().reset_index(), hue = 'Genre', ax = ax)


# ### 3) Actors Analysis

# In[ ]:





# In[143]:


# Checking the Top 10 actors with most number of movies
actors_with_movies_counts = dict( sorted(dict(Counter(df_movie_analysis.Actors.tolist())).items(), key=operator.itemgetter(1),reverse=True))
for i, (actors, num_movies) in enumerate(actors_with_movies_counts.items()):
    if i > 10: break
    print('{actors} had acted in {movies} movies.'.format(actors=actors, movies=num_movies))


# In[156]:


actors_group = df_movie_analysis.groupby(['Actors', 'Genre'])
actors_favorite_Genre = actors_group['imdbID'].count().reset_index()
actors_favorite_Genre.columns = ['Actors', 'Genre', 'Number_of_Movies']
actors_favorite_Genre.sort_values(by=['Number_of_Movies'], ascending=False, inplace=True)
actors_favorite_Genre[:10].sort_values(by=['Genre'], ascending=False).head(10)


# In[ ]:





# ### 4) Writer Analysis

# In[138]:


# Checking the Top 10 Writer with most number of movies
writer_with_movies_counts = dict( sorted(dict(Counter(df_movie_analysis.Person.tolist())).items(), key=operator.itemgetter(1),reverse=True))
for i, (writer, num_movies) in enumerate(writer_with_movies_counts.items()):
    if i > 10: break
    print('{writer} had written {movies} movies.'.format(writer=writer, movies=num_movies))


# In[139]:


movies_writer_group = df_movie_analysis.groupby(['Person', 'Genre'])
witer_favorite_Genre = movies_writer_group['imdbID'].count().reset_index()
witer_favorite_Genre.columns = ['Writer', 'Genre', 'Number_of_Movies']
witer_favorite_Genre.sort_values(by=['Number_of_Movies'], ascending=False, inplace=True)
witer_favorite_Genre[:10].sort_values(by=['Genre'], ascending=False).head(10)


# In[ ]:





# In[ ]:





# In[ ]:





# ## Creating the Connection to the sqlite database.

# In[88]:


# Creating the connection to the database.
conn = sqlite3.connect('database/movies_database.db')


# In[89]:


# creating the cursor for executing the queries.
cursor = conn.cursor()


# In[ ]:





# In[ ]:





# In[90]:


# closing the cursor connection.
cursor.close()


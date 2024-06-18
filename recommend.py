import numpy as np
import pandas as pd
import ast
import nltk
import pickle
movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')
movies = movies.merge(credits, on = 'title')
movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]
movies.head()
#checking for missing data
#movies.isnull().sum()
#for removing missing data
movies.dropna(inplace = True)
#for checking is any duplicated data
#movies.duplicated().sum()
def convert(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L
movies['genres']=movies['genres'].apply(convert)
movies['keywords']=movies['keywords'].apply(convert)
def convert3(obj):
    L = []
    counter = 0
    for i in ast.literal_eval(obj):
        if counter!=3:
            L.append(i['name'])
            counter+=1
        else:
            break
    return L
movies['cast']=movies['cast'].apply(convert3)
def fetch_director(obj):
    L = []
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            L.append(i['name'])
    return L
movies['crew']=movies['crew'].apply(fetch_director)
movies['overview']=movies['overview'].apply(lambda x:x.split())

movies['genres']=movies['genres'].apply(lambda x:[i.replace(" ","") for i in x])
movies['keywords']=movies['keywords'].apply(lambda x:[i.replace(" ","") for i in x])
movies['cast']=movies['cast'].apply(lambda x:[i.replace(" ","") for i in x])
movies['crew']=movies['crew'].apply(lambda x:[i.replace(" ","") for i in x])

movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] +movies['crew']

#creating new movies dataframe using only title id and tags
newmovies = movies[['movie_id','title','tags']]
newmovies['tags']=newmovies['tags'].apply(lambda x:" ".join(x))
newmovies['tags']=newmovies['tags'].apply(lambda x:x.lower())


#vectorization ' converting tags string into vectors ' using libraries form scikit
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 5000, stop_words = 'english')
vectors = cv.fit_transform(newmovies['tags']).toarray()
cv.get_feature_names_out()

from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
def stem(text):
    y =[]
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)    

newmovies['tags']=newmovies['tags'].apply(stem)
cv.get_feature_names_out()
from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity(vectors)
similarity


#main function for recommending movies
def recommend(movie):
    movie_index = newmovies[newmovies['title']==movie].index[0]
    distance = similarity[movie_index]
    movie_list = sorted(list(enumerate(distance)),reverse=True,key=lambda x:x[1])[1:6]
    for i in movie_list:
        print(newmovies.iloc[i[0]].title)

recommend('Batman Begins')


#for frontend part to upload files to app.py we used pickle
pickle.dump(newmovies,open('movies.pkl','wb'))
pickle.dump(newmovies.to_dict(),open('moviesdict.pkl','wb'))
pickle.dump(similarity,open('similarity.pkl','wb'))
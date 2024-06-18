import streamlit as st
import pickle
import pandas as pd

def recommend(movie):
    movie_index = movies[movies['title']==movie].index[0]
    distance = similarity[movie_index]
    movie_list = sorted(list(enumerate(distance)),reverse=True,key=lambda x:x[1])[1:6]

    recommended = []
    for i in movie_list:
        recommended.append(movies.iloc[i[0]].title)
    return recommended

movies_dict = pickle.load(open('moviesdict.pkl','rb'))
movies = pd.DataFrame(movies_dict)
similarity = pickle.load(open('similarity.pkl','rb'))
st.title('Movie Recommender System')
option = st.selectbox(
    "How would you like to be contacted?",
    movies['title'].values)

st.write("You selected:", option)

if st.button("Recommend"):
    recommendation = recommend(option)
    for i in recommendation:
        st.write(i)



        #for running the app , entre "streamlit run app.py" in terminal
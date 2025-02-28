# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 20:08:15 2025

@author: lenovo
"""

import numpy as np
import pickle
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
movies_data = pd.read_csv('movies.csv')
# replacing the null values with null string
selected_features = ['genres', 'keywords', 'tagline', 'cast', 'director']
for feature in selected_features:
  movies_data[feature] = movies_data[feature].fillna('')
combined_features = movies_data['genres']+' '+movies_data['keywords']+' '+movies_data['tagline']+' '+movies_data['cast']+' '+movies_data['director']
feature_vectors = vectorizer.fit_transform(combined_features)
similarity = cosine_similarity(feature_vectors)

# loading the saved model
loaded_model = pickle.load(open('C:/Users/lenovo/Downloads/Movie Recommendation System web app/trained_model.sav', 'rb'))

def movie_prediction(input_data):
    list_of_all_titles = movies_data['title'].tolist()
    find_close_match = loaded_model(input_data, list_of_all_titles)
    close_match = find_close_match[0]
    
    index_of_the_movie = movies_data[movies_data.title == close_match]['index'].values[0]
    similarity_score = list(enumerate(similarity[index_of_the_movie]))
    sorted_similar_movies = sorted(similarity_score, key=lambda x: x[1], reverse=True)

    # Store movie recommendations in a list
    recommended_movies = ["Movies suggested for you:"]
    
    i = 1
    for movie in sorted_similar_movies:
        index = movie[0]
        title_from_index = movies_data.loc[movies_data.index == index, 'title'].values[0]
        if i <= 20:  # Limit to 20 movies
            recommended_movies.append(f"{i}. {title_from_index}")
            i += 1

    return "\n".join(recommended_movies)  # Return the list as a formatted string

        
def main():
      
    # giving a title
    st.title('Movie Prediction Web App')
    
    # getting the input data from the user
    #Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age
    
    movie_name = st.text_input('Enter your favourite movie name:')
    
    # code for Prediction
    diagnosis = ''
    
    # creating a button for Prediction
    
    if st.button('Predict Movie'):
        diagnosis = movie_prediction(movie_name)
    
    st.success(diagnosis)
    
    
if __name__ == '__main__':
    main()
    

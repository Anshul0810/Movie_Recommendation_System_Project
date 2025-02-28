# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 21:07:33 2025

@author: lenovo
"""

import numpy as np
import pickle
import pandas as pd
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

movie_name = input(' Enter your favourite movie name : ')

movies_data = pd.read_csv('movies.csv')

list_of_all_titles = movies_data['title'].tolist()

find_close_match = loaded_model(movie_name, list_of_all_titles)

close_match = find_close_match[0]

index_of_the_movie = movies_data[movies_data.title == close_match]['index'].values[0]

similarity_score = list(enumerate(similarity[index_of_the_movie]))

sorted_similar_movies = sorted(similarity_score, key = lambda x:x[1], reverse = True)

print('Movies suggested for you : \n')

i=1
for movie in sorted_similar_movies:
  index = movie[0]
  title_from_index = movies_data[movies_data.index == index]['title'].values[0]
  if(i<30):
    print(i, '-',title_from_index)
    i+=1
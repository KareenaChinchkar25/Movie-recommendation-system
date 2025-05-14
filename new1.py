import pandas as pd
import numpy as np
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

# Load datasets
credits = pd.read_csv('tmdb_5000_credits.csv')
movies = pd.read_csv('tmdb_5000_movies.csv')

# Merge the datasets on 'title'
movies = movies.merge(credits, left_on='title', right_on='title')

# Function to convert string representation of list to a list of names
def convert(obj):
    return [i['name'] for i in ast.literal_eval(obj)]

# Apply the conversion functions
movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)
movies['cast'] = movies['cast'].apply(lambda x: [i['name'] for i in ast.literal_eval(x)[:3]])  # Top 3 actors
movies['crew'] = movies['crew'].apply(lambda x: [i['name'] for i in ast.literal_eval(x) if i['job'] == 'Director'])

# Create a 'tags' column combining genres, keywords, cast, and crew
movies['tags'] = movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
movies['tags'] = movies['tags'].apply(lambda x: " ".join(x))
movies['tags'] = movies['tags'].apply(lambda x: x.lower())  # Convert to lowercase

# Create the TF-IDF matrix and cosine similarity matrix
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['tags'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Function to get movie recommendations based on cosine similarity
def get_recommendations(title, cosine_sim=cosine_sim):
    idx = movies[movies['title'] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]  # Get top 10 similar movies
    movie_indices = [i[0] for i in sim_scores]
    return movies['title'].iloc[movie_indices]

# Test the recommendation function
print(get_recommendations('The Dark Knight Rises'))

# Save the movies and cosine similarity matrix using pickle
with open('movie_data.pkl', 'wb') as file:
    pickle.dump((movies, cosine_sim), file)

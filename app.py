from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Load the movie dataset and similarity matrix
movies = joblib.load('movie_joblib.pkl')
similarity = joblib.load('similarity_joblib.pkl')

# Function to recommend movies
def recommend(movie):
    index = movies[movies['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    recommended_movies = [movies.iloc[i[0]].title for i in distances[1:6]]
    return recommended_movies

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend_movies():
    movie_name = request.form['movie']
    recommendations = recommend(movie_name)
    return jsonify(recommendations)

if __name__ == '__main__':
    app.run(debug=True)

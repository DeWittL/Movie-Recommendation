# Importing required libraries
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

# Load the datasets
ratings = pd.read_csv('ratings.csv')  # Load ratings data
movies = pd.read_csv('movies.csv')    # Load movies data

# Merging datasets
data = pd.merge(ratings, movies, on='movieId')

# Preprocessing
data['userId'] = data['userId'].astype('category').cat.codes  # Encoding user IDs
data['movieId'] = data['movieId'].astype('category').cat.codes  # Encoding movie IDs

# Create user and movie ID mappings
num_users = data['userId'].nunique()
num_movies = data['movieId'].nunique()

# Prepare input and output data for training
X = data[['userId', 'movieId']].values
y = data['rating'].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the Movie Recommendation Model
class MovieRecommendationModel(tf.keras.Model):
    def __init__(self, num_users, num_movies, embedding_size):
        super(MovieRecommendationModel, self).__init__()
        self.user_embedding = layers.Embedding(num_users, embedding_size)
        self.movie_embedding = layers.Embedding(num_movies, embedding_size)
        self.flatten = layers.Flatten()

    def call(self, inputs):
        user_vector = self.user_embedding(inputs[0])
        movie_vector = self.movie_embedding(inputs[1])
        dot_product = tf.reduce_sum(user_vector * movie_vector, axis=1)
        return dot_product

# Initialize the model
embedding_size = 50
model = MovieRecommendationModel(num_users, num_movies, embedding_size)

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit([X_train[:, 0], X_train[:, 1]], y_train, epochs=10, batch_size=64)

# Make predictions
predictions = model.predict([X_test[:, 0], X_test[:, 1]])

# Output predicted ratings
predicted_ratings = pd.DataFrame({'UserId': X_test[:, 0], 'MovieId': X_test[:, 1], 'PredictedRating': predictions.flatten()})

# Display the first few predicted ratings
print(predicted_ratings.head())


import pandas as pd
import traceback
import time
import numpy as np

from keras.models import load_model
from datetime import datetime, timedelta
from .db import run_query, engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import create_engine
from sqlalchemy import text


db_config = {
    'user': 'root',
    'password': 'root',
    'host': 'localhost',
    'database': 'rec_sys',
}

# Create a connection to the database
engine = create_engine(
    f"mysql+pymysql://{db_config['user']}:{db_config['password']}@{db_config['host']}/{db_config['database']}")


class Recommender:
    def __init__(self):
        self.model_path = r'D:\Users\vitor\RecSys_Course\keras_recommendation_model.h5'
        self.loaded_model = load_model(self.model_path)

    def get_recommendations_from_cache(self, user_id):
        query = f"""
        SELECT movie_id, predicted_rating
        FROM recommendations_cache
        WHERE user_id = {user_id} AND timestamp > '{(datetime.now() - timedelta(days=10)).strftime('%Y-%m-%d %H:%M:%S')}'
        """
        df = run_query(query)
        return df if not df.empty else None

    def store_recommendations_in_cache(self, recommendations):
        if recommendations:
            user_id = recommendations[0][0]

            with engine.connect() as conn:
                with conn.begin():
                    # Use the text() function to create an executable SQL expression
                    delete_query = text(
                        "DELETE FROM recommendations_cache WHERE user_id = :user_id")
                    conn.execute(delete_query, {'user_id': user_id})

                    df = pd.DataFrame(recommendations, columns=[
                                      'user_id', 'movie_id', 'predicted_rating'])
                    df['timestamp'] = datetime.now()
                    df.to_sql('recommendations_cache', con=conn,
                              if_exists='append', index=False)

    def get_recommendations(self, user_id, batch_size=10000):
        watched_query = f"SELECT movieId FROM ratings WHERE userId = {user_id}"
        watched_df = run_query(watched_query)
        watched_movie_ids = set(watched_df['movieId'].values)

        cached_recommendations = self.get_recommendations_from_cache(user_id)
        if cached_recommendations is not None and not cached_recommendations.empty:
            # Filter out watched movies from cached recommendations
            not_watched = cached_recommendations[~cached_recommendations['movie_id'].isin(
                watched_movie_ids)]
            print(not_watched)
            if len(not_watched) == 15:
                return not_watched.sort_values(by='predicted_rating', ascending=False).head(15)

        query = f"SELECT * FROM edited_rating WHERE userId = {user_id}"
        recommendations = []
        offset = 0
        max_movie_index = 26887
        while True:
            df_user = run_query(
                query + f" LIMIT {batch_size} OFFSET {offset}")

            if df_user.empty:
                break

            user_idx = df_user['userId'].values
            movie_idx = df_user['movieId'].values
            valid_idx = (movie_idx < max_movie_index)
            user_idx = user_idx[valid_idx]
            movie_idx = movie_idx[valid_idx]

            if len(user_idx) == 0:
                break

            preds = self.loaded_model.predict(
                [user_idx, movie_idx]).flatten()
            print(type(preds))
            valid_movie_ids = df_user['movieId'].values[valid_idx]
            for movie_id, pred in zip(valid_movie_ids, preds):
                if movie_id not in watched_movie_ids:  # Exclude movies that the user has already watched
                    recommendations.append((user_id, movie_id, pred))

            offset += batch_size

            recommendations.sort(key=lambda x: x[2], reverse=True)
            # Store only recommendations for movies not already watched
            top_recommendations = [(user_id, movie_id, pred) for user_id, movie_id,
                                   pred in recommendations if movie_id not in watched_movie_ids][:15]
            
            self.store_recommendations_in_cache(top_recommendations)
            print(top_recommendations)
            return top_recommendations

    def get_movie_details(self, movie_ids):
        query = f"SELECT * FROM movie WHERE movieId IN {tuple(movie_ids)}"
        df_movies = run_query(query)
        return df_movies.to_dict(orient='records')

    def set_movie_rating(self, user_id, movie_id, rating):
        print(
            f"Inserting user_id: {user_id}, movie_id: {movie_id}, rating: {rating} into pending ratings table")

        movie_idx_query = "SELECT movie_idx FROM edited_rating WHERE movieId = %s LIMIT 1"
        csv_file_path = 'machine_learning_examples/recommenders/dataset-ml-latest-small/edited_rating.csv'

        try:
            with engine.connect() as conn:
                with conn.begin():
                    movie_idx_result = pd.read_sql_query(
                        movie_idx_query, con=conn, params=(movie_id,))

                    print(movie_idx_result)

                    if not movie_idx_result.empty:
                        movie_idx = movie_idx_result.iloc[0]['movie_idx']
                        edited_query = text(
                            "INSERT INTO edited_rating (userId, movieId, movie_idx, rating) VALUES (:user_id, :movie_id, :movie_idx, :rating)")
                        rating_query = text(
                            "INSERT INTO ratings (userId, movieId, rating) VALUES (:user_id, :movie_id, :rating)")
                        conn.execute(rating_query, {
                                     'user_id': user_id, 'movie_id': movie_id, 'rating': rating})
                        conn.execute(edited_query, {
                                     'user_id': user_id, 'movie_id': movie_id, 'movie_idx': movie_idx, 'rating': rating})
                        print("Rating in set_movie_rating: ", rating)
                        new_rating_row = pd.DataFrame({
                            'userId': [user_id],
                            'movieId': [movie_id],
                            'rating': [float(rating)],
                            'movie_idx': [movie_idx]
                        })

                        # Append to CSV without the index
                        new_rating_row.to_csv(
                            csv_file_path, mode='a', header=False, index=False)
                        return True
                    else:
                        print(f"No movie_idx found for movieId: {movie_id}")
                        return False
        except Exception as e:
            print(f"An error occurred: {e}")
            traceback.print_exc()  # This will print the full traceback
            return False


    def calculate_rmse(self, user_id):

        predicted_ratings = pd.read_csv('predicted_ratings_test_cf.csv')
        
        user_predictions = predicted_ratings[predicted_ratings['user_id'] == user_id]

        if user_predictions.empty:
            print(f'No predictions found for user {user_id}')
            return None
        
        predictions = user_predictions['predicted_rating']
        actual_ratings = user_predictions['actual_rating']
        # Calculate squared differences
        squared_diff = (predictions - actual_ratings) ** 2

        # Calculate mean squared error
        mse = np.mean(squared_diff)

        # Calculate RMSE
        rmse = np.sqrt(mse)
        print(f'RMSE: {rmse}')
        return rmse

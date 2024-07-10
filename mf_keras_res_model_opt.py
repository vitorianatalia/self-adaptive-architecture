import pandas as pd
import time
from datetime import datetime, timedelta
from sqlalchemy import create_engine
from keras.models import load_model

start_time = time.time()
db_config = {
    "user": "root",
    "password": "root",
    "host": "localhost",
    "database": "rec_sys",
}

engine = create_engine(
    f"mysql+pymysql://{db_config['user']}:{db_config['password']}@{db_config['host']}/{db_config['database']}"
)

model_path = r"D:\Users\vitor\RecSys_Course\keras_recommendation_model_smaller.h5"
loaded_model = load_model(model_path)


def get_recommendations_from_cache(user_id):
    query = f"""
    SELECT movie_id, predicted_rating
    FROM recommendations_cache
    WHERE user_id = {user_id} AND timestamp > '{(datetime.now() - timedelta(days=10)).strftime('%Y-%m-%d %H:%M:%S')}'
    """
    df = pd.read_sql_query(query, con=engine)
    return df if not df.empty else None


def store_recommendations_in_cache(recommendations):
    df = pd.DataFrame(
        recommendations, columns=["user_id", "movie_id", "predicted_rating"]
    )
    df["timestamp"] = datetime.now()
    df.to_sql("recommendations_cache", con=engine, if_exists="append", index=False)


def get_movie_details(movie_ids):
    query = f"SELECT * FROM movie WHERE movieId IN {tuple(movie_ids)}"
    df_movies = pd.read_sql_query(query, con=engine)
    return df_movies.to_dict(orient="records")


print("Time elapsed:", (time.time() - start_time))

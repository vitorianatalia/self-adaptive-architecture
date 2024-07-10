import pandas as pd
import os

current_dir = os.path.dirname(os.path.realpath(__file__))
rating_file_path = os.path.join(
    current_dir, "..", "..", "large_files", "movielens-20m-dataset", "rating.csv"
)
edited_rating_file_path = os.path.join(
    current_dir, "..", "..", "large_files", "movielens-20m-dataset", "edited_rating.csv"
)

df = pd.read_csv(rating_file_path)

df.userId = df.userId - 1

unique_movie_ids = set(df.movieId.values)
movie2idx = {}
count = 0
for movie_id in unique_movie_ids:
    movie2idx[movie_id] = count
    count += 1

df["movie_idx"] = df.apply(lambda row: movie2idx[row.movieId], axis=1)

df = df.drop(columns=["timestamp"])


df.to_csv(edited_rating_file_path, index=False)

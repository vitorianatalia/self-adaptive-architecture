import pandas as pd

movie_csv_path = r"C:\Users\vitor\Downloads\Recommender-System-on-MovieLens-dataset-main\Recommender-System-on-MovieLens-dataset-main\movielens-20m-dataset\movie.csv"  # Replace with the actual file path
movie_df = pd.read_csv(movie_csv_path)

unique_genres = set()
for genres in movie_df["genres"].str.split("|"):
    unique_genres.update(genres)
unique_genres.discard("(no genres listed)")

columns = [
    "movie id",
    "movie title",
    "release date",
    "video release date",
    "IMDb URL",
] + list(unique_genres)
new_df = pd.DataFrame(columns=columns)

for _, row in movie_df.iterrows():
    movie_id = row["movieId"]
    movie_title = row["title"]
    genres = set(row["genres"].split("|"))
    genres.discard("(no genres listed)")

    row_dict = {
        "movie id": movie_id,
        "movie title": movie_title,
        "release date": "",
        "video release date": "",
        "IMDb URL": "",
    }
    row_dict.update({genre: 1 if genre in genres else 0 for genre in unique_genres})

    new_df = pd.concat([new_df, pd.DataFrame([row_dict])], ignore_index=True)

output_csv_path = "transformed_movie.csv"
new_df.to_csv(output_csv_path, index=False, sep=",")

print(f"Transformed data saved to {output_csv_path}")

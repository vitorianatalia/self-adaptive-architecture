import numpy as np
import pandas as pd
import time
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sqlalchemy import create_engine

db_config = {
    "user": "root",
    "password": "root",
    "host": "localhost",
    "database": "rec_sys",
}
engine = create_engine(
    f"mysql+pymysql://{db_config['user']}:{db_config['password']}@{db_config['host']}/{db_config['database']}?charset=utf8mb4",
    echo=False,
)

start_time = time.time()


def refresh_list_movies_seen():
    csv_file_path = r"D:\Users\vitor\RecSys_Course\machine_learning_examples\recommenders\dataset-ml-latest-small\edited_rating.csv"
    column_names1 = ["user id", "movie id", "rating", "timestamp"]
    return pd.read_csv(csv_file_path, sep=",", header=0, names=column_names1)


dataset = refresh_list_movies_seen()

d = "movie id | movie title | release date | video release date | IMDb URL | Action | Adventure | Animation | Children | Comedy | Crime | Documentary | Drama | Fantasy | Film-Noir | Horror | IMAX | Musical | Mystery | Romance | Sci-Fi | Thriller | War | Western"
column_names2 = d.split(" | ")

transf_csv = r"D:\Users\vitor\RecSys_Course\machine_learning_examples\recommenders\dataset-ml-latest-small\transformed_movie.csv"

items_dataset = pd.read_csv(
    transf_csv, sep=",", header=0, names=column_names2, encoding="latin-1"
)

movie_dataset = items_dataset[["movie id", "movie title"]]


merged_dataset = pd.merge(
    dataset, items_dataset[["movie id", "movie title"]], on="movie id", how="inner"
)
merged_dataset[
    (merged_dataset["movie title"] == "Chasing Amy (1997)")
    & (merged_dataset["user id"] == 894)
]
refined_dataset = merged_dataset.groupby(
    by=["user id", "movie title"], as_index=False
).agg({"rating": "mean"})


num_users = len(refined_dataset["user id"].value_counts())
num_items = len(refined_dataset["movie title"].value_counts())
print("Unique number of users in the dataset: {}".format(num_users))
print("Unique number of movies in the dataset: {}".format(num_items))

rating_count_df = pd.DataFrame(
    refined_dataset.groupby(["rating"]).size(), columns=["count"]
)

ax = (
    rating_count_df.reset_index()
    .rename(columns={"index": "rating score"})
    .plot(
        "rating",
        "count",
        "bar",
        figsize=(12, 8),
        title="Count for Each Rating Score",
        fontsize=12,
    )
)

ax.set_xlabel("movie rating score")
ax.set_ylabel("number of ratings")

total_count = num_items * num_users
zero_count = total_count - refined_dataset.shape[0]

zero_count_df = pd.DataFrame({"count": zero_count}, index=[0.0])
rating_count_df = pd.concat(
    [rating_count_df, zero_count_df], ignore_index=True, verify_integrity=True
).sort_index()

rating_count_df["log_count"] = np.log(rating_count_df["count"])

rating_count_df = rating_count_df.reset_index().rename(
    columns={"index": "rating score"}
)

ax = rating_count_df.plot(
    "rating score",
    "log_count",
    "bar",
    figsize=(12, 8),
    title="Count for Each Rating Score (in Log Scale)",
    logy=True,
    fontsize=12,
)

ax.set_xlabel("movie rating score")
ax.set_ylabel("number of ratings")


movies_count_df = pd.DataFrame(
    refined_dataset.groupby("movie title").size(), columns=["count"]
)

ax = (
    movies_count_df.sort_values("count", ascending=False)
    .reset_index(drop=True)
    .plot(figsize=(12, 8), title="Rating Frequency of All Movies", fontsize=12)
)
ax.set_xlabel("movie Id")
ax.set_ylabel("number of ratings")

user_to_movie_df = refined_dataset.pivot(
    index="user id", columns="movie title", values="rating"
).fillna(0)


user_to_movie_sparse_df = csr_matrix(user_to_movie_df.values)
user_to_movie_dense_df = pd.DataFrame(user_to_movie_sparse_df.toarray())


knn_model = NearestNeighbors(metric="cosine", algorithm="brute")
knn_model.fit(user_to_movie_sparse_df)


def updateDatabase():
    dataset = refresh_list_movies_seen()

    d = "movie id | movie title | release date | video release date | IMDb URL | Action | Adventure | Animation | Children | Comedy | Crime | Documentary | Drama | Fantasy | Film-Noir | Horror | IMAX | Musical | Mystery | Romance | Sci-Fi | Thriller | War | Western"
    column_names2 = d.split(" | ")
    transf_csv = r"D:\Users\vitor\RecSys_Course\machine_learning_examples\recommenders\dataset-ml-latest-small\transformed_movie.csv"
    items_dataset = pd.read_csv(
        transf_csv, sep=",", header=0, names=column_names2, encoding="latin-1"
    )

    merged_dataset = pd.merge(
        dataset, items_dataset[["movie id", "movie title"]], on="movie id", how="inner"
    )
    merged_dataset[
        (merged_dataset["movie title"] == "Chasing Amy (1997)")
        & (merged_dataset["user id"] == 894)
    ]
    print("Merged dataset:", merged_dataset[merged_dataset["user id"] == 11])
    refined_dataset = merged_dataset.groupby(
        by=["user id", "movie title"], as_index=False
    ).agg({"rating": "mean"})

    num_users = len(refined_dataset["user id"].value_counts())
    num_items = len(refined_dataset["movie title"].value_counts())
    print("Unique number of users in the dataset: {}".format(num_users))
    print("Unique number of movies in the dataset: {}".format(num_items))

    rating_count_df = pd.DataFrame(
        refined_dataset.groupby(["rating"]).size(), columns=["count"]
    )

    ax = (
        rating_count_df.reset_index()
        .rename(columns={"index": "rating score"})
        .plot(
            "rating",
            "count",
            "bar",
            figsize=(12, 8),
            title="Count for Each Rating Score",
            fontsize=12,
        )
    )

    ax.set_xlabel("movie rating score")
    ax.set_ylabel("number of ratings")

    total_count = num_items * num_users
    zero_count = total_count - refined_dataset.shape[0]

    zero_count_df = pd.DataFrame({"count": zero_count}, index=[0.0])
    rating_count_df = pd.concat(
        [rating_count_df, zero_count_df], ignore_index=True, verify_integrity=True
    ).sort_index()

    rating_count_df["log_count"] = np.log(rating_count_df["count"])

    rating_count_df = rating_count_df.reset_index().rename(
        columns={"index": "rating score"}
    )

    ax = rating_count_df.plot(
        "rating score",
        "log_count",
        "bar",
        figsize=(12, 8),
        title="Count for Each Rating Score (in Log Scale)",
        logy=True,
        fontsize=12,
    )

    ax.set_xlabel("movie rating score")
    ax.set_ylabel("number of ratings")

    movies_count_df = pd.DataFrame(
        refined_dataset.groupby("movie title").size(), columns=["count"]
    )

    ax = (
        movies_count_df.sort_values("count", ascending=False)
        .reset_index(drop=True)
        .plot(figsize=(12, 8), title="Rating Frequency of All Movies", fontsize=12)
    )
    ax.set_xlabel("movie Id")
    ax.set_ylabel("number of ratings")

    user_to_movie_df = refined_dataset.pivot(
        index="user id", columns="movie title", values="rating"
    ).fillna(0)

    return user_to_movie_df


def get_similar_users(user, n=5):

    knn_input = np.asarray([user_to_movie_df.values[user - 1]])
    distances, indices = knn_model.kneighbors(knn_input, n_neighbors=n + 1)

    print("Top", n, "users who are very much similar to the User-", user, "are: ")
    print(" ")
    for i in range(1, len(distances[0])):
        print(
            i, ". User:", indices[0][i] + 1, "separated by distance of", distances[0][i]
        )
    return indices.flatten()[1:] + 1, distances.flatten()[1:]


user_id = 104
similar_user_list, distance_list = get_similar_users(user_id, 5)

weightage_list = distance_list / np.sum(distance_list)


mov_rtngs_sim_users = user_to_movie_df.values[similar_user_list]


def list_movies_seen(user_id):

    user_to_movie_df = updateDatabase()
    user_ratings = user_to_movie_df.loc[user_id]
    watched_movies = user_ratings[user_ratings > 0]
    print("Movies seen by the User:", user_id)
    print(" ")
    print(watched_movies)

    seen_movies_ratings = []
    for title in watched_movies.index:
        movie_id = movie_dataset[movie_dataset["movie title"] == title][
            "movie id"
        ].iloc[0]
        rating = watched_movies[title]
        print("Movie:", title, "Movie ID:", movie_id, "Rating:", rating)
        seen_movies_ratings.append([movie_id, rating])

    return seen_movies_ratings


movies_list = user_to_movie_df.columns


weightage_list = weightage_list[:, np.newaxis] + np.zeros(len(movies_list))
weightage_list.shape

new_rating_matrix = weightage_list * mov_rtngs_sim_users
mean_rating_list = new_rating_matrix.sum(axis=0)


def recommend_movies(n):
    n = min(len(mean_rating_list), n)


recommend_movies(10)


def filtered_movie_recommendations(n):
    zero_indices = np.where(mean_rating_list == 0)[0]
    if zero_indices.size > 0:
        first_zero_index = zero_indices[-1]
        sorted_index = np.argsort(mean_rating_list)[::-1]
        sorted_index = sorted_index[: sorted_index.tolist().index(first_zero_index)]
    else:
        sorted_index = np.argsort(mean_rating_list)[::-1]

    n = min(len(sorted_index), n)
    movies_watched = list(
        refined_dataset[refined_dataset["user id"] == user_id]["movie title"]
    )
    filtered_movie_list = [
        movies_list[i] for i in sorted_index if movies_list[i] not in movies_watched
    ][:n]

    if len(filtered_movie_list) == 0:
        print(
            "There are no new movies left to recommend. Try increasing the number of similar users to consider."
        )
    else:
        return filtered_movie_list


def recommender_system(user_id, n_similar_users=5, n_movies=10):
    def get_similar_users(user, n=5):

        knn_input = np.asarray([user_to_movie_df.values[user - 1]])

        distances, indices = knn_model.kneighbors(knn_input, n_neighbors=n + 1)

        print("Top", n, "users who are very much similar to the User-", user, "are: ")
        print(" ")

        for i in range(1, len(distances[0])):
            print(
                i,
                ". User:",
                indices[0][i] + 1,
                "separated by distance of",
                distances[0][i],
            )
        print("")
        return indices.flatten()[1:] + 1, distances.flatten()[1:]

    def filtered_movie_recommendations(n=10):

        first_zero_index = np.where(mean_rating_list == 0)[0][-1]
        sortd_index = np.argsort(mean_rating_list)[::-1]
        sortd_index = sortd_index[: list(sortd_index).index(first_zero_index)]
        n = min(len(sortd_index), n)
        movies_watched = list(
            refined_dataset[refined_dataset["user id"] == user_id]["movie title"]
        )
        filtered_movie_list = list(movies_list[sortd_index])
        count = 0
        final_movie_list = []
        recommended_movie_ids = []
        for i in filtered_movie_list:
            if i not in movies_watched:
                count += 1
                final_movie_list.append(i)
            if count == n:
                break
        if count == 0:
            return "There are no movies left which are not seen by the input users and seen by similar users. May be increasing the number of similar users who are to be considered may give a chance of suggesting an unseen good movie."
        else:
            recommended_movie_ids = movie_dataset[
                movie_dataset["movie title"].isin(final_movie_list)
            ]["movie id"].tolist()
            return recommended_movie_ids

    similar_user_list, distance_list = get_similar_users(user_id, n_similar_users)
    weightage_list = distance_list / np.sum(distance_list)
    mov_rtngs_sim_users = user_to_movie_df.values[similar_user_list]
    movies_list = user_to_movie_df.columns
    weightage_list = weightage_list[:, np.newaxis] + np.zeros(len(movies_list))
    new_rating_matrix = weightage_list * mov_rtngs_sim_users
    mean_rating_list = new_rating_matrix.sum(axis=0)
    print("")
    print("Movies recommended based on similar users are: ")
    print("")
    recommended_movie_ids = filtered_movie_recommendations(n_movies)
    return recommended_movie_ids


movie_to_user_df = refined_dataset.pivot(
    index="movie title", columns="user id", values="rating"
).fillna(0)

movie_to_user_df.head()

movie_to_user_sparse_df = csr_matrix(movie_to_user_df.values)
movie_to_user_sparse_df

movies_list = list(movie_to_user_df.index)
movies_list[:10]

movie_dict = {movie: index for index, movie in enumerate(movies_list)}

case_insensitive_movies_list = [i.lower() for i in movies_list]

knn_movie_model = NearestNeighbors(metric="cosine", algorithm="brute")
knn_movie_model.fit(movie_to_user_sparse_df)


def get_similar_movies(movie, n=10):
    index = movie_dict[movie]
    knn_input = np.asarray([movie_to_user_df.values[index]])
    n = min(len(movies_list) - 1, n)
    distances, indices = knn_movie_model.kneighbors(knn_input, n_neighbors=n + 1)


movie_name = "101 Dalmatians (1996)"

get_similar_movies(movie_name, 15)


def get_possible_movies(movie):

    temp = ""
    possible_movies = case_insensitive_movies_list.copy()
    for i in movie:
        out = []
        temp += i
        for j in possible_movies:
            if temp in j:
                out.append(j)
        if len(out) == 0:
            return possible_movies
        out.sort()
        possible_movies = out.copy()

    return possible_movies


class invalid(Exception):
    pass


def spell_correction():

    try:

        movie_name = input("Enter the Movie name: ")
        movie_name_lower = movie_name.lower()
        if movie_name_lower not in case_insensitive_movies_list:
            raise invalid
        else:
            num_recom = int(input("Enter Number of movie recommendations needed: "))
            get_similar_movies(
                movies_list[case_insensitive_movies_list.index(movie_name_lower)],
                num_recom,
            )

    except invalid:

        possible_movies = get_possible_movies(movie_name_lower)

        if len(possible_movies) == len(movies_list):
            print("Movie name entered is does not exist in the list ")
        else:
            indices = [case_insensitive_movies_list.index(i) for i in possible_movies]
            print(
                "Entered Movie name is not matching with any movie from the dataset . Please check the below suggestions :\n",
                [movies_list[i] for i in indices],
            )
            spell_correction()


num_entries = movie_to_user_df.shape[0] * movie_to_user_df.shape[1]
num_zeros = (movie_to_user_df == 0).sum(axis=1).sum()
ratio_zeros = num_zeros / num_entries
print("There is about {:.2%} of ratings in our data is missing".format(ratio_zeros))

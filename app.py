from flask import Flask, send_from_directory, request, jsonify
from mf_keras_res_model_opt import get_movie_details
from KNN_recommendations import (
    recommender_system,
    list_movies_seen,
    refresh_list_movies_seen,
)
from refactor.recommendations import Recommender
from flask_cors import CORS
import sys
import time
import pandas as pd

sys.path.append(
    r"C:\Users\vitor\OneDrive\Área de Trabalho\RecSys_Course\machine_learning_examples\recommenders"
)


app = Flask(__name__, static_folder="frontend-app/build")
recommender = Recommender()

CORS(app, origins="http://localhost:3000")


@app.route("/")
def serve():
    return send_from_directory(app.static_folder, "index.html")


rmse_threshold = 0.7


@app.route("/keras-get-recommendations", methods=["POST"])
def get_recommendations_endpoint():
    start_time = time.time()
    data = request.json
    if not data or "user_id" not in data:
        return jsonify({"error": "User ID is required."}), 400

    try:
        user_id = int(data["user_id"])
    except ValueError:
        return jsonify({"error": "User ID must be an integer."}), 400

    recommended_movies = recommender.get_recommendations(user_id)

    if recommended_movies is None or len(recommended_movies) == 0:
        return jsonify({"error": "No recommendations available."}), 404

    if isinstance(recommended_movies, pd.DataFrame):
        movie_ids = recommended_movies["movie_id"].tolist()
    else:
        movie_ids = [movie_id for _, movie_id, _ in recommended_movies]

    movie_details = recommender.get_movie_details(movie_ids)
    print("Time taken: ", time.time() - start_time)
    return jsonify({"payload": movie_details})


@app.route("/knn-recommendations", methods=["POST"])
def knn_recommendations():
    data = request.get_json()
    if not data or "user_id" not in data:
        return jsonify({"error": "User ID is required."}), 400

    try:
        user_id = int(data["user_id"])
    except ValueError:
        return jsonify({"error": "User ID must be an integer."}), 400

    n = data.get("n", 15)
    predictions = recommender_system(user_id, n_similar_users=n, n_movies=n)
    if predictions is None or len(predictions) == 0:
        return jsonify({"error": "No recommendations available."}), 404

    movie_details_list = get_movie_details(predictions)

    recommendations = []
    for detail in movie_details_list:
        movie_dict = {
            "movieId": detail["movieId"],
            "title": detail["title"],
            "genres": detail["genres"],
        }
        recommendations.append(movie_dict)

    return jsonify({"payload": recommendations})


@app.route("/list-seen", methods=["POST"])
def list_seen():
    data = request.get_json()
    if not data or "user_id" not in data:
        return jsonify({"error": "User ID is required."}), 400

    try:
        user_id = int(data["user_id"])
    except ValueError:
        return jsonify({"error": "User ID must be an integer."}), 400

    refresh_list_movies_seen()
    seen_movies = list_movies_seen(user_id)
    print("Seen movies: ", seen_movies)
    movie_ids = [movie[0] for movie in seen_movies]
    print("Movie IDs: ", movie_ids)
    movie_details_list = get_movie_details(movie_ids)
    for details in movie_details_list:
        movie_id = details["movieId"]
        rating = next((item[1] for item in seen_movies if item[0] == movie_id), None)
        details["rating"] = rating

    if seen_movies is None or len(seen_movies) == 0:
        return jsonify({"error": "No movies found."}), 404
    print("Movie details: ", movie_details_list)
    return jsonify(movie_details_list)


@app.route("/update-rating", methods=["POST"])
def update_rating():
    data = request.get_json()
    print(data)
    if not data or not all(k in data for k in ("user_id", "movie_id", "rating")):
        return jsonify({"error": "User ID, Movie ID, and Rating are required."}), 400

    user_id = data["user_id"]
    movie_id = data["movie_id"]
    rating = data["rating"]
    print("user_id: ", user_id, "movie_id: ", movie_id, "rating: ", rating)
    success = recommender.set_movie_rating(user_id, movie_id, rating)
    get_recommendations()
    if success:
        return jsonify({"message": "Rating received and will be processed."}), 200
    else:
        return jsonify({"error": "Failed to store the rating."}), 500


@app.route("/get-recommendations", methods=["POST"])
def get_recommendations():
    data = request.get_json()
    if not data or "user_id" not in data:
        return jsonify({"error": "User ID is required."}), 400
    try:
        user_id = int(data["user_id"])
    except ValueError:
        return jsonify({"error": "User ID must be an integer."}), 400
    algorithm = ""
    rmse = recommender.calculate_rmse(user_id)

    if rmse >= rmse_threshold:
        # KERAS
        start_time = time.time()
        recommended_movies = recommender.get_recommendations(user_id)

        if recommended_movies is None or len(recommended_movies) == 0:
            return jsonify({"error": "No recommendations available."}), 404

        if isinstance(recommended_movies, pd.DataFrame):
            movie_ids = recommended_movies["movie_id"].tolist()
        else:
            movie_ids = [movie_id for _, movie_id, _ in recommended_movies]

        movie_details = recommender.get_movie_details(movie_ids)
        print("Time taken for Keras: ", time.time() - start_time)

        algorithm = "Keras"
        return jsonify(
            {
                "settings": {"algorithm": algorithm, "rmse": rmse},
                "payload": movie_details,
            }
        )
    else:
        # KNN
        start_time = time.time()
        n = data.get("n", 15)
        predictions = recommender_system(user_id, n_similar_users=n, n_movies=n)
        if predictions is None or len(predictions) == 0:
            return jsonify({"error": "No recommendations available."}), 404

        movie_details_list = get_movie_details(predictions)

        recommendations = []
        for detail in movie_details_list:
            movie_dict = {
                "movieId": detail["movieId"],
                "title": detail["title"],
                "genres": detail["genres"],
            }
            recommendations.append(movie_dict)

        print(movie_dict["movieId"])
        print("Time taken for KNN: ", time.time() - start_time)
        algorithm = "KNN"
        return jsonify(
            {
                "settings": {"algorithm": algorithm, "rmse": rmse},
                "payload": recommendations,
            }
        )


if __name__ == "__main__":
    app.run(use_reloader=True, port=5000, threaded=True)

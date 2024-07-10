import React, { useState } from "react";
import MovieTable from "./MovieTable";
import "./styles.css"; 
import "./styles/Recommendations.css";

function Recommendations() {
  const [userId, setUserId] = useState("");
  const [recommendations, setRecommendations] = useState(null);
  const [error, setError] = useState("");
  const [buttonClicked, setButtonClicked] = useState(false);
  const [recommendationType, setRecommendationType] = useState("auto");
  const [seenMovies, setSeenMovies] = useState(null);
  const [modelName, setModelName] = useState("");
  const [rmse, setRmse] = useState("");

  const fetchRecommendations = async () => {
    let endpoint = "";

    switch (recommendationType) {
      case "auto":
        endpoint = "http://127.0.0.1:5000/get-recommendations";
        break;
      case "keras":
        endpoint = "http://127.0.0.1:5000/keras-get-recommendations";
        break;
      case "cf":
        endpoint = "http://127.0.0.1:5000/knn-recommendations";
        break;
      case "seen":
        endpoint = "http://127.0.0.1:5000/list-seen";
        break;
      default:
        setError("Invalid recommendation type selected.");
        return;
    }
    setButtonClicked(true);

    try {
      const response = await fetch(endpoint, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ user_id: userId }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || "Server error");
      }

      const data = await response.json();
      if (recommendationType === "seen") {
        setSeenMovies(data);
        setRecommendations(null);
      } else {
        if (data.settings && data.settings.algorithm == "Keras") {
          setModelName("Keras");
          setRmse(1 - data.settings.rmse);
          setRecommendations(data.payload);
        } else if (data.settings && data.settings.algorithm == "KNN") {
          setModelName("KNN");
          setRecommendations(data.payload);
          setRmse(1 - data.settings.rmse);
        } else {
          setRecommendations(data.payload);
        }
        setSeenMovies(null);
      }
      setError("");
    } catch (error) {
      console.error(`Error fetching ${recommendationType}:`, error);
      setError(error.message);
      setRecommendations(null);
      setSeenMovies(null);
    }
  };

  const handleOptionChange = (option) => {
    setRecommendationType(option);
  };

  return (
    <div className="container">
      <h1>Movie Recommendations</h1>

      <div className="options">
        <label>
          <input
            type="radio"
            value="auto"
            checked={recommendationType === "auto"}
            onChange={() => handleOptionChange("auto")}
          />
          Auto Recommendations
        </label>
        <label>
          <input
            type="radio"
            value="keras"
            checked={recommendationType === "keras"}
            onChange={() => handleOptionChange("keras")}
          />
          Keras Recommendations
        </label>
        <label>
          <input
            type="radio"
            value="cf"
            checked={recommendationType === "cf"}
            onChange={() => handleOptionChange("cf")}
          />
          CF Recommendations
        </label>
        <label>
          <input
            type="radio"
            value="seen"
            checked={recommendationType === "seen"}
            onChange={() => handleOptionChange("seen")}
          />
          Seen Movies
        </label>
      </div>

      <input
        type="text"
        value={userId}
        onChange={(e) => setUserId(e.target.value)}
        placeholder="Enter user ID"
        className="user-id-input"
      />

      <button
        onClick={fetchRecommendations}
        className="get-recommendations-button"
      >
        {recommendationType === "seen"
          ? "Get Seen Movies"
          : "Get Recommendations"}
      </button>
      {recommendationType === "auto" && buttonClicked && (
        <div className="model-info">
          <p>Model: <span className="model-name">{modelName}</span></p>
          <p>RMSE: <span className={rmse < 0.3 ? 'rmse-green' : 'rmse-red'}>{rmse}</span></p>
        </div>
      )}
      {recommendations && (
        
        <div className="seen-movies-container">
          <MovieTable
            movies={recommendations}
            showRatings={true}
            userId={userId}
          />
        </div>
      )}
      {seenMovies && (
        <div className="seen-movies-container">
          <MovieTable movies={seenMovies} showRatings={true} userId={userId} />
        </div>
      )}
    </div>
  );
}

export default Recommendations;

import React, { useState, useEffect } from "react";
import { FaStar } from "react-icons/fa";
import "./styles.css"; 

function StarRating({ rating: initialRating, userId, movieId }) {
  const [rating, setRating] = useState(initialRating); 
  const [hover, setHover] = useState(null);

  useEffect(() => {
    setRating(initialRating);
  }, [initialRating]);

  const handleRatingChange = async (rating) => {
    console.log(`New rating for movie ${movieId}: ${rating}`);

    try {
      const response = await fetch("http://127.0.0.1:5000/update-rating", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ user_id: userId, movie_id: movieId, rating }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || "Server error updating rating");
      }

      console.log("Rating updated successfully");
      document.getElementsByClassName('get-recommendations-button')[0].click()    } catch (error) {
      console.error("Error updating rating:", error);
    }
  };


  return (
    <div className="star-rating">
      {[...Array(5)].map((star, index) => {
        const ratingValue = index + 1;
        return (
          <label key={ratingValue} className="star-container">
            <input
              type="radio"
              name="rating"
              value={ratingValue}
              onClick={() => handleRatingChange(ratingValue)}
              className="star-radio"
            />
            <FaStar
              className="star"
              size={20}
              color={ratingValue <= (hover || rating) ? "#ffc107" : "#e4e5e9"}
              onMouseEnter={() => setHover(ratingValue)}
              onMouseLeave={() => setHover(null)}
            />
          </label>
        );
      })}
    </div>
  );
}

export default StarRating;

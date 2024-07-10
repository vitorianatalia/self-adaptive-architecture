import React from "react";
import StarRating from "./StarRating"; 

const MovieTable = ({ movies, showRatings, userId }) => {
  return (
    <table className="movie-table">
      <thead>
        <tr>
          <th>ID</th>
          <th>Title</th>
          <th>Genres</th>
          {showRatings && <th>Rating</th>}
        </tr>
      </thead>
      <tbody>
        {movies.map((movie) => (
          <tr key={movie.movieId}>
            <td>{movie.movieId}</td>
            <td className="movie-title">{movie.title}</td>
            <td>
              {movie.genres.split("|").map((genre, index) => (
                <p key={index} className={`genre ${genre.toLowerCase()}`}>
                  {genre}
                </p>
              ))}
            </td>{" "}
            {showRatings && (
              <td>
                <StarRating
                  rating={movie.rating}
                  userId={userId}
                  movieId={movie.movieId}
                />
              </td>
            )}
          </tr>
        ))}
      </tbody>
    </table>
  );
};

export default MovieTable;

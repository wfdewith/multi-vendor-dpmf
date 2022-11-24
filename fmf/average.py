from collections import Counter

import numpy as np
import pandas as pd
from numpy.random import RandomState
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

TRAIN_SIZE = 0.9
TOTAL_MEAN_BUDGET = 0.01

ML10M_GENRES = {
    "Action",
    "Adventure",
    "Animation",
    "Children",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Fantasy",
    "Film-Noir",
    "Horror",
    "Musical",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Thriller",
    "War",
    "Western",
}

RNG = RandomState(1432)


def vertical(ratings, min_rating, max_rating, movie_splits, epsilon):
    ratings_means_splits = []
    ratings_test_splits = []

    for movie_split in movie_splits:
        ratings_split_movie_index = pd.Index(movie_split, name="movie")
        ratings_split = ratings[ratings["movie"].isin(movie_split)]
        ratings_split_train, ratings_split_test = train_test_split(
            ratings_split, train_size=TRAIN_SIZE, random_state=RNG
        )
        ratings_test_splits.append(ratings_split_test)

        total_ratings_sum = ratings_split_train["rating"].sum()
        total_ratings_count = ratings_split_train["rating"].count()

        total_ratings_s = (
            total_ratings_sum
            - total_ratings_count * (min_rating + max_rating) / 2
            + RNG.laplace(0, (max_rating - min_rating) / (TOTAL_MEAN_BUDGET * epsilon))
        )
        total_ratings_c = total_ratings_count + RNG.laplace(
            0, 2 / (TOTAL_MEAN_BUDGET * epsilon)
        )
        if total_ratings_c <= 1:
            total_ratings_mean_priv = (min_rating + max_rating) / 2
        else:
            total_ratings_mean_priv = (
                total_ratings_s / total_ratings_c + (min_rating + max_rating) / 2
            )

        ratings_sums = pd.Series(
            data=0, index=ratings_split_movie_index, dtype=np.float64
        )
        ratings_counts = pd.Series(
            data=0, index=ratings_split_movie_index, dtype=np.float64
        )
        ratings_sums = ratings_sums.add(
            ratings_split_train.groupby("movie")["rating"].sum(), fill_value=0
        )
        ratings_counts = ratings_counts.add(
            ratings_split_train.groupby("movie")["rating"].count(), fill_value=0
        )

        ratings_sums[ratings_sums == 0] += total_ratings_mean_priv
        ratings_counts[ratings_counts == 0] += 1

        ratings_s = (
            ratings_sums
            - ratings_counts * (min_rating + max_rating) / 2
            + RNG.laplace(
                0,
                (max_rating - min_rating) / ((1 - TOTAL_MEAN_BUDGET) * epsilon),
                size=len(ratings_sums),
            )
        )
        ratings_c = ratings_counts + RNG.laplace(
            0, 2 / ((1 - TOTAL_MEAN_BUDGET) * epsilon), size=len(ratings_counts)
        )
        ratings_split_means = pd.Series(
            data=0, index=ratings_split_movie_index, dtype=np.float64
        )
        ratings_split_means = ratings_s / ratings_c + (min_rating + max_rating) / 2
        ratings_split_means[ratings_c <= 1] = (min_rating + max_rating) / 2
        ratings_means_splits.append(ratings_split_means)

    ratings_test = pd.concat(ratings_test_splits)
    ratings_means = pd.concat(ratings_means_splits)
    ratings_means.clip(min_rating, max_rating, inplace=True)

    ratings_test = ratings_test.merge(
        ratings_means.rename("rating_pred"), how="left", on="movie"
    )
    return mean_squared_error(ratings_test["rating"], ratings_test["rating_pred"], squared=False)


print("Reading ratings.dat")
ratings: pd.DataFrame = pd.read_csv(
    "ratings.dat",
    sep="::",
    names=("user", "movie", "rating", "timestamp"),
)  # type: ignore

print("Ratings:")
print(ratings.head())

users = ratings["user"].unique()
movies = ratings["movie"].unique()

max_rating = ratings["rating"].max()
min_rating = ratings["rating"].min()
print()

print("Reading movies.dat")
movie_genres = {}
with open("movies.dat", "r") as f_avg:
    for line in f_avg:
        movie_id, _, genres = line.strip().split("::")
        genres = genres.split("|")
        genres = [genre for genre in genres if genre in ML10M_GENRES]
        if genres:
            movie_genres[int(movie_id)] = genres

genre_counts = Counter(genre for genres in movie_genres.values() for genre in genres)
movie_genre = {
    movie: min(genres, key=lambda genre: genre_counts[genre])
    for movie, genres in movie_genres.items()
}
genres = {}
for movie, genre in movie_genre.items():
    genres.setdefault(genre, []).append(movie)

epsilons = (0.037, 0.055, 0.086, 0.174)

with open("averages_genre.csv", "w") as f_avg, open("averages_random.csv", "w") as f_rnd:
    f_avg.write("vendors,epsilon,rmse\n")
    f_rnd.write("vendors,epsilon,rmse\n")

    genre_movie_splits = genres.values()
    for epsilon in epsilons:
        rmses = []
        print(
            f"Per genre vertical s={len(genres)} (epsilon={epsilon}): ", end="", flush=True
        )
        for i in range(10):
            rmses.append(
                vertical(
                    ratings,
                    min_rating,
                    max_rating,
                    genre_movie_splits,
                    epsilon,
                )
            )
            print(i, end="", flush=True)
        rmse = np.mean(rmses)
        f_avg.write(f"18 (genre),{epsilon},{rmse}\n")
        print("\nResult:", rmse)

    for vendors in (2, 5, 10, 18):
        RNG.shuffle(movies)
        movie_splits = np.array_split(movies, vendors)
        for epsilon in epsilons:
            rmses = []
            print(f"Random vertical s={vendors} (epsilon={epsilon}): ", end="", flush=True)
            for i in range(10):
                rmses.append(
                    vertical(ratings, min_rating, max_rating, movie_splits, epsilon)
                )
                print(i, end="", flush=True)
            rmse = np.mean(rmses)
            if vendors == 18:
                f_avg.write(f"18 (random),{epsilon},{rmse}\n")
            else:
                f_rnd.write(f"{vendors},{epsilon},{rmse}\n")
            print("\nResult:", rmse)

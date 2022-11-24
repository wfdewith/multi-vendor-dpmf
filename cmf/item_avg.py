from collections import Counter

import numpy as np
import pandas as pd
from numpy.random import RandomState
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

TRAIN_SIZE = 0.8
TOTAL_MEAN_BUDGET = 0.1

RNG = RandomState(1009)


def dp_global_average(ratings, epsilon):
    max_rating = ratings["rating"].max()
    min_rating = ratings["rating"].min()

    ratings_train, ratings_test = train_test_split(
        ratings, train_size=TRAIN_SIZE, random_state=RNG
    )

    total_ratings_sum = ratings_train["rating"].sum()
    total_ratings_count = ratings_train["rating"].count()
    total_ratings_s = (
        total_ratings_sum
        - total_ratings_count * (min_rating + max_rating) / 2
        + RNG.laplace(0, (max_rating - min_rating) / epsilon)
    )
    total_ratings_c = total_ratings_count + RNG.laplace(0, 2 / epsilon)
    if total_ratings_c <= 1:
        total_ratings_mean = (min_rating + max_rating) / 2
    else:
        total_ratings_mean = (
            total_ratings_s / total_ratings_c + (min_rating + max_rating) / 2
        )

    ratings_test["rating_pred"] = total_ratings_mean

    return mean_squared_error(
        ratings_test["rating"], ratings_test["rating_pred"], squared=False
    )


def dp_item_averages(ratings, epsilon):
    movies = ratings["movie"].unique()
    movie_index = pd.Index(movies, name="movie")

    max_rating = ratings["rating"].max()
    min_rating = ratings["rating"].min()

    ratings_train, ratings_test = train_test_split(
        ratings, train_size=TRAIN_SIZE, random_state=RNG
    )

    total_ratings_sum = ratings_train["rating"].sum()
    total_ratings_count = ratings_train["rating"].count()
    total_ratings_s = (
        total_ratings_sum
        - total_ratings_count * (min_rating + max_rating) / 2
        + RNG.laplace(0, (max_rating - min_rating) / (TOTAL_MEAN_BUDGET * epsilon))
    )
    total_ratings_c = total_ratings_count + RNG.laplace(
        0, 2 / (TOTAL_MEAN_BUDGET * epsilon)
    )
    if total_ratings_c <= 1:
        total_ratings_mean = (min_rating + max_rating) / 2
    else:
        total_ratings_mean = (
            total_ratings_s / total_ratings_c + (min_rating + max_rating) / 2
        )

    ratings_sums = pd.Series(data=0, index=movie_index, dtype=np.float64)
    ratings_counts = pd.Series(data=0, index=movie_index, dtype=np.float64)

    ratings_sums = ratings_sums.add(
        ratings_train.groupby("movie")["rating"].sum(), fill_value=0
    )
    ratings_counts = ratings_counts.add(
        ratings_train.groupby("movie")["rating"].count(), fill_value=0
    )

    ratings_sums[ratings_sums == 0] += total_ratings_mean
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
    ratings_means = ratings_s / ratings_c + (min_rating + max_rating) / 2
    ratings_means[ratings_c <= 1] = (min_rating + max_rating) / 2

    ratings_means = ratings_sums / ratings_counts
    ratings_means.clip(min_rating, max_rating, inplace=True)

    ratings_test = ratings_test.merge(
        ratings_means.rename("rating_pred"), how="left", on="movie"
    )
    return mean_squared_error(
        ratings_test["rating"], ratings_test["rating_pred"], squared=False
    )


print("Reading ratings.dat")
ratings: pd.DataFrame = pd.read_csv(
    "ratings.dat",
    sep="::",
    names=("user", "movie", "rating", "timestamp"),
)  # type: ignore

print("Ratings:")
print(ratings.head())

print()

epsilon = 0.1

global_average = []
item_averages = []

for i in range(10):
    global_average.append(dp_global_average(ratings, epsilon))
    item_averages.append(dp_item_averages(ratings, epsilon))

print(np.mean(global_average))
print(np.mean(item_averages))


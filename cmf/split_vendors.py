import os

import numpy as np
import pandas as pd
from numpy.random import default_rng
from sklearn.model_selection import train_test_split

RNG = default_rng()


def circ(l, a, b):
    assert a < b
    assert a < len(l)
    if b >= len(l):
        return l[a:] + l[: b % len(l)]
    else:
        return l[a:b]


def create_mapper(user_ids, movie_ids):
    def libfm_mapper(row):
        rating = row["rating"]
        user_id = user_ids[row["user"]]
        movie_id = movie_ids[row["movie"]]
        return f"{rating} {user_id}:1 {movie_id}:1\n"

    return libfm_mapper


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

RNG.shuffle(users)

ratings_splits = [
    ratings[ratings["user"].isin(user_split)]
    for user_split in np.array_split(users, 10)
]
train_test_splits = [
    train_test_split(ratings_split, train_size=0.8) for ratings_split in ratings_splits
]

# HORIZONTAL
for n in range(1, 11):
    print(f"{n} parties (horizontal)")
    for i in range(10):
        print(f"Party {i}")
        os.makedirs(f"data_vendors/r{n}", exist_ok=True)

        train_split = pd.concat(tr for tr, _ in circ(train_test_splits, i, i + n))
        test_split = train_test_splits[i][1]

        ratings_split = pd.concat((train_split, test_split))

        user_split = ratings_split["user"].unique()
        movie_split = ratings_split["movie"].unique()
        user_ids = {u: j for j, u in enumerate(user_split)}
        movie_ids = {m: j + len(user_split) for j, m in enumerate(movie_split)}
        mapper = create_mapper(user_ids, movie_ids)

        train_split = train_split.apply(mapper, axis=1)
        test_split = test_split.apply(mapper, axis=1)

        with open(f"data_vendors/r{n}/{i}.train", "w") as tr:
            tr.writelines(train_split)
        with open(f"data_vendors/r{n}/{i}.test", "w") as ts:
            ts.writelines(test_split)
        with open(f"data_vendors/r{n}/{i}.group", "w") as g:
            g.writelines("0\n" for _ in range(len(user_ids)))
            g.writelines("1\n" for _ in range(len(movie_ids)))

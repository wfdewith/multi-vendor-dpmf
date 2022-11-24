import os

import numpy as np
import pandas as pd
from numpy.random import RandomState
from sklearn.model_selection import train_test_split

RNG = RandomState(48437)
TRAIN_SIZE = 0.9


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

# HORIZONTAL
for n in (10, 40):
    print(f"{n} parties (horizontal)")
    RNG.shuffle(users)
    user_splits = np.array_split(users, n)
    for i, user_split in enumerate(user_splits):
        os.makedirs(f"data/r{n}", exist_ok=True)
        print(f"{i+1}/{n}")
        ratings_split = ratings[ratings["user"].isin(user_split)]

        movie_split = ratings_split["movie"].unique()
        user_ids = {u: j for j, u in enumerate(user_split)}
        movie_ids = {m: j + len(user_split) for j, m in enumerate(movie_split)}
        mapper = create_mapper(user_ids, movie_ids)

        ratings_split = ratings_split.apply(mapper, axis=1)

        ratings_split_train, ratings_split_test = train_test_split(
            ratings_split, train_size=TRAIN_SIZE, random_state=RNG
        )

        with open(f"data/r{n}/{i}.train", "w") as tr:
            tr.writelines(ratings_split_train)
        with open(f"data/r{n}/{i}.test", "w") as ts:
            ts.writelines(ratings_split_test)
        with open(f"data/r{n}/{i}.group", "w") as g:
            g.writelines("0\n" for _ in range(len(user_ids)))
            g.writelines("1\n" for _ in range(len(movie_ids)))

import os

import numpy as np
import pandas as pd
from numpy.random import RandomState
from sklearn.model_selection import KFold

RNG = RandomState(34214)

def create_mapper(user_ids, movie_ids):
    def libfm_mapper(row):
        rating = row["rating"]
        user_id = user_ids[row["user"]]
        movie_id = movie_ids[row["movie"]]
        return f"{rating} {user_id}:1 {movie_id}:1\n"

    return libfm_mapper


print("Reading u.data")
ratings: pd.DataFrame = pd.read_csv(
    "u.data",
    sep="\t",
    names=("user", "movie", "rating", "timestamp"),
)  # type: ignore

print("Ratings:")
print(ratings.head())

users = ratings["user"].unique()
movies = ratings["movie"].unique()

# HORIZONTAL
for n in range(1, 11):
    print(f"{n} parties (horizontal)")
    RNG.shuffle(users)
    user_splits = np.array_split(users, n)
    for i, user_split in enumerate(user_splits):
        os.makedirs(f"data/r{n}/p{i}", exist_ok=True)
        print(f"{i+1}/{n}")
        user_split = set(user_split)

        ratings_split = ratings[ratings["user"].isin(user_split)]

        movie_split = ratings_split["movie"].unique()
        user_ids = {u: j for j, u in enumerate(user_split)}
        movie_ids = {m: j + len(user_split) for j, m in enumerate(movie_split)}
        mapper = create_mapper(user_ids, movie_ids)

        ratings_split = ratings_split.apply(mapper, axis=1)

        kf = KFold(n_splits=10, shuffle=True)
        for split_i, (train_index, test_index) in enumerate(kf.split(ratings_split)):
            ratings_split_train = ratings_split.iloc[train_index]
            ratings_split_test = ratings_split.iloc[test_index]

            with open(f"data/r{n}/p{i}/{split_i}.train", "w") as tr:
                tr.writelines(ratings_split_train)
            with open(f"data/r{n}/p{i}/{split_i}.test", "w") as ts:
                ts.writelines(ratings_split_test)
            with open(f"data/r{n}/p{i}/{split_i}.group", "w") as g:
                g.writelines("0\n" for _ in range(len(user_ids)))
                g.writelines("1\n" for _ in range(len(movie_ids)))

import os

import pandas as pd
from numpy.random import RandomState
from sklearn.model_selection import train_test_split

RNG = RandomState(1221)

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

os.makedirs(f"data_full", exist_ok=True)
user_ids = {u: j for j, u in enumerate(users)}
movie_ids = {m: j + len(users) for j, m in enumerate(movies)}
mapper = create_mapper(user_ids, movie_ids)

ratings = ratings.apply(mapper, axis=1)

ratings_train, ratings_test = train_test_split(ratings, train_size=0.8, random_state=RNG)

with open(f"data_full/0.train", "w") as tr:
    tr.writelines(ratings_train)
with open(f"data_full/0.test", "w") as ts:
    ts.writelines(ratings_test)
with open(f"data_full/0.group", "w") as g:
    g.writelines("0\n" for _ in range(len(user_ids)))
    g.writelines("1\n" for _ in range(len(movie_ids)))

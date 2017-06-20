import sys
import pandas as pd

# OPEN 5000 IMDB MOVIE DATASET
df = pd.read_csv("C:/Users/Tom/Documents/Informatiekunde/Thesis/data/movie_metadata.csv")

# CREATE LIST OF DIRECTORS
directorlist = df.director_name.unique().astype(str).tolist()

with open("C:/Users/Tom/Documents/Informatiekunde/Thesis/Feature sets/directors.txt", "w") as directorfile:
    for director in directorlist:
        directorfile.write(director.lower() + "\n")

# CREATE LIST OF ACTORS
actors1 = df.actor_1_name.unique().astype(str).tolist()
actors2 = df.actor_2_name.unique().astype(str).tolist()
actors3 = df.actor_3_name.unique().astype(str).tolist()
actorlist = list(set(actors1 + actors2 + actors3))

with open("C:/Users/Tom/Documents/Informatiekunde/Thesis/Feature sets/actors.txt", "w") as actorfile:
    for actor in actorlist:
       actorfile.write(actor.lower() + "\n")

# CREATE LIST OF TITLES
titlelist = df.movie_title.astype(str).tolist()

with open("C:/Users/Tom/Documents/Informatiekunde/Thesis/features/titles.txt", "w") as titlefile:
    for title in titlelist:
        titlefile.write(title.lower() + "\n")

#CREATE LIST OF GENRES
genrelist = df.genres.unique().astype(str).tolist()

with open("C:/Users/Tom/Documents/Informatiekunde/Thesis/Feature sets/genres.txt", "w") as genrefile:
    for genre in genrelist:
       genrefile.write(genre.lower() + "\n")

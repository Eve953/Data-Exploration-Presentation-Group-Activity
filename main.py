import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('Data-Exploration-Presentation-Group-Activity/movies/imdb_top_movies_1980_2026.csv')

# STEPHANIE AND ISAURO: which movies have most votes and does that it affect its average_rating?
higher_num_votes = df[df["num_votes"] > 500000]

display(higher_num_votes.nlargest(20, "num_votes")[["title", "num_votes", "average_rating"]])


display(higher_num_votes.nlargest(20, "average_rating")[["title", "num_votes", "average_rating"]])


display(df["num_votes"].corr(df["average_rating"]))

display(higher_num_votes["num_votes"].corr(higher_num_votes["average_rating"]))


fig, ax = plt.subplots(2,2, figsize=(16, 10))
ax[0][0].scatter(df["num_votes"], df["average_rating"])
m, b = np.polyfit(df["num_votes"], df["average_rating"], 1)
ax[0][0].plot(df["num_votes"], m * df["num_votes"] + b, color="red", linestyle="--")
ax[0][0].set_ylim(0, 10)
ax[0][0].set_title("Relationship between Number of Votes and Average Rating")
ax[0][0].set_xlabel("Number of Votes (in millions)")
ax[0][0].set_ylabel("Average Rating")

sns.heatmap(
    df[["num_votes", "average_rating", "runtime_minutes"]].corr(),
    annot=True, 
    cmap="coolwarm",
    linewidths=0.5,
    ax=ax[0][1]
)
ax[0][1].set_title("Correlation between Number of Votes and Average Rating")


ax[1][0].scatter(higher_num_votes["num_votes"], higher_num_votes["average_rating"])
m, b = np.polyfit(higher_num_votes["num_votes"], higher_num_votes["average_rating"], 1)
ax[1][0].plot(higher_num_votes["num_votes"], m * higher_num_votes["num_votes"] + b, color="red", linestyle="--")
ax[1][0].set_ylim(0, 10)
ax[1][0].set_title("Relationship between Number of Votes and Average Rating")
ax[1][0].set_xlabel("Number of Votes (in millions)")
ax[1][0].set_ylabel("Average Rating")


sns.heatmap(higher_num_votes[["num_votes", "average_rating", "runtime_minutes"]].corr(),
annot=True, 
cmap="coolwarm", 
linewidths=0.5,
ax=ax[1][1])
ax[1][1].set_title("Correlation between Number of Votes and Average Rating")




# ROBERT AND MATTHEW: trends of most popular genres per year

df[["year", "genres"]]
genre_rating_df = df[["year", "genres", "average_rating"]].copy()

genre_rating_df = genre_rating_df.dropna(subset = ["genres", "average_rating"])

genre_rating_df["decade"] = (genre_rating_df["year"] // 10) * 10

genre_rating_df["genres"] = genre_rating_df["genres"].str.split(",")

genre_rating_df = genre_rating_df.explode("genres")

genre_rating_df["genres"] = genre_rating_df["genres"].str.strip()
# print(genre_rating_df)

#avg rating for each genre in each decade
avg_ratings = (
    genre_rating_df.groupby(["decade", "genres"])["average_rating"]
    .mean()
    .reset_index()
)

avg_ratings["average_rating"] = avg_ratings["average_rating"].round(1)


# Get all decades
decades = sorted(avg_ratings["decade"].unique())

# Get the top 5 genres for each decade
top_5_list = []

for decade in decades:
    decade_data = avg_ratings[avg_ratings["decade"] == decade]
    top_5 = decade_data.sort_values("average_rating", ascending=False).head(5)
    top_5_list.append(top_5)

# Combine all top 5 results into one DataFrame
top_5_all = pd.concat(top_5_list)

# Pivot the data so decades are rows and genres are columns
pivot_df = top_5_all.pivot_table(
    index="decade",
    columns="genres",
    values="average_rating",
    fill_value=0
)

# Plot stacked bar chart
ax = pivot_df.plot(
    kind="bar",
    stacked=True,
    figsize=(12, 7)
)

plt.title("Top 5 Highest-Rated Genres by Decade")
plt.xlabel("Decade")
plt.ylabel("Stacked Average IMDb Ratings")
plt.xticks(rotation=0)
plt.legend(title="Genre", bbox_to_anchor=(1.05, 1), loc="upper left")

plt.tight_layout()
plt.show()

# ALEX AND EVELYN: Movie Rating Trends over time
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

grouped = df.groupby("year")
x = []
y_list_rating = []

for year in range(1980, 2027):
    x.append(year)
    y_list_rating.append(grouped.get_group(year)["average_rating"])

y_rating = []
for rating in y_list_rating:
    y_rating.append(np.average(rating))

ax1.plot(x, y_rating)
ax1.set_xlabel("Year")
ax1.set_ylabel("Average Rating")
ax1.set_title("Movie Rating Trend Over Time")

y_list_runtime = []

for year in range(1980, 2027):
    y_list_runtime.append(grouped.get_group(year)["runtime_minutes"])

y_runtime = []
for runtime in y_list_runtime:
    y_runtime.append(np.average(runtime))

ax2.plot(x, y_runtime)

ax2.set_xlabel("Year")
ax2.set_ylabel("Average Runtime (in minutes)")
ax2.set_title("Average Movie Runtime per Year")

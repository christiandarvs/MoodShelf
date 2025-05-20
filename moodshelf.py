# %% [markdown]
# # MoodShelf

# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option("display.max_columns", None)

df = pd.read_csv("./goodreads-data-50k.csv")
df.head()

# %% [markdown]
# ## Explanatory Data Analysis

# %%
cols_to_keep = [
    "title",
    "author",
    "rating",
    "description",
    "language",
    "genres",
    "pages",
    "publisher",
    "publishDate",
    "numRatings",
    "likedPercent",
    "coverImg",
]

df_subset = df[cols_to_keep]
df_subset.head()

# %%
duplicate_groups = df_subset[df_subset.duplicated(subset=cols_to_keep, keep=False)]

num_duplicate_groups = duplicate_groups.duplicated(subset=cols_to_keep).sum()

print(f"Number of duplicate groups: {num_duplicate_groups}")

# %%
# View the first few duplicate rows
duplicate_groups.sort_values(by=cols_to_keep).head()

# %%
# Remove duplicates and keep the first occurrence
df_subset_cleaned = df_subset.drop_duplicates(subset=cols_to_keep, keep="first")

print(f"Before: {df_subset.shape}")
print(f"After: {df_subset_cleaned.shape}")


# %%
def create_histogram(x, xlabel, title):
    plt.figure(figsize=(8, 5))
    plt.hist(df_subset_cleaned[x], bins=20, edgecolor="black")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.grid(axis="y", alpha=0.7)
    plt.tight_layout()
    plt.show()


# %%
create_histogram(x="rating", xlabel="Rating", title="Book Rating Distribution")

# %%
create_histogram(
    x="likedPercent", xlabel="Liked Percent", title="Book Liked Percent Distribution"
)

# %% [markdown]
# ## Preprocess the Description

# %%
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download("punkt_tab")

stop_words = set(stopwords.words("english"))


def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = word_tokenize(text)  # tokenize
    filtered_tokens = [word for word in tokens if word not in stop_words]
    cleaned_text = " ".join(filtered_tokens)
    return cleaned_text


# %%
df_subset_cleaned["description_cleaned"] = df_subset_cleaned["description"].apply(
    preprocess_text
)

# %%
df_subset_cleaned.head()

# %%
"""
Return books that have:
    Ratings: 4 - 5
    Liked Percentage: 90% - 100%
    Number of Ratings: >= 250,000
"""

high_rating_and_percent = df_subset_cleaned[
    (df_subset_cleaned["rating"] >= 4)
    & (df_subset_cleaned["rating"] <= 5)
    & (df_subset_cleaned["likedPercent"] >= 90)
    & (df_subset_cleaned["likedPercent"] <= 100)
    & (df_subset_cleaned["numRatings"] >= 150_000)
]

high_rating_and_percent.sort_values(
    ascending=[False, False], by=["rating", "likedPercent"]
)
high_rating_and_percent.reset_index(drop=True)
high_rating_and_percent.to_csv("highly_rated_popular_books.csv", index=False)

# %% [markdown]
# ## Emotion Mapping

# %%
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "j-hartmann/emotion-english-distilroberta-base"

# Use RobertaTokenizer and RobertaForSequenceClassification, NOT BertTokenizer
tokenizer = RobertaTokenizer.from_pretrained(model_name)
model = RobertaForSequenceClassification.from_pretrained(model_name)
model = model.to(device)


# Get labels
emotion_labels = [model.config.id2label[i] for i in range(model.config.num_labels)]
print("Model's Actual Emotion Labels:", emotion_labels)


# %%
def classify_emotions_in_batch(texts, batch_size=32):
    results = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]

        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)

        for prob in probs:
            # Get index of max probability
            max_idx = prob.argmax().item()
            predicted_label = emotion_labels[max_idx]
            results.append(predicted_label)
    return results


# %%
high_rating_and_percent["description"] = high_rating_and_percent["description"].astype(
    str
)

# Run classification
high_rating_and_percent["predicted_emotion"] = classify_emotions_in_batch(
    high_rating_and_percent["description"].tolist()
)

# Save output
high_rating_and_percent.to_csv(
    "highly_rated_popular_books_with_emotions.csv", index=False
)
print("Saved to highly_rated_popular_books_with_emotions.csv")

# %%
df = pd.read_csv("./highly_rated_popular_books_with_emotions.csv")
df

# %%
emotion_counts = df["predicted_emotion"].value_counts()

# Plot with correct label order
plt.pie(emotion_counts, labels=emotion_counts.index, autopct="%1.1f%%")
plt.axis("equal")
plt.title("Predicted Emotions from Book Descriptions")
plt.show()

# Print raw counts
print(emotion_counts)

# %%
print(
    f'The Shining Description: {df[df["title"] == "The Shining"].description_cleaned.values}'
)

print(
    f'\nThe Shining Prediction Emotion: {df[df["title"] == "The Shining"].predicted_emotion}'
)

# %%
book_emotions = (
    df.groupby("title")["predicted_emotion"]
    .value_counts(normalize=True)
    .unstack()
    .fillna(0)
)


book_emotions = book_emotions.reindex(columns=emotion_labels, fill_value=0)

print("Emotional Fingerprints:")
display(book_emotions)

# %%
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def recommend_books(user_emotion_vector, book_emotions_df, full_df, top_n=10):
    book_vectors = book_emotions_df[emotion_labels].values
    similarities = cosine_similarity([user_emotion_vector], book_vectors)

    similar_books_idx = np.argsort(similarities[0])[::-1]
    top_indices = similar_books_idx[:top_n]
    top_titles = book_emotions_df.index[top_indices]

    # Filter the original DataFrame and reset index
    recommended_books_df = (
        full_df[full_df["title"].isin(top_titles)]
        .drop_duplicates(subset="title")
        .reset_index(drop=True)
    )

    # Sort by recommendation order
    recommended_books_df["title_order"] = recommended_books_df["title"].apply(
        lambda x: list(top_titles).index(x)
    )
    recommended_books_df = (
        recommended_books_df.sort_values("title_order")
        .drop(columns="title_order")
        .reset_index(drop=True)
    )

    return recommended_books_df


# Example user emotion vector
user_input = [
    1.0,  # anger
    0.0,  # disgust
    1.0,  # fear
    0.0,  # joy
    0.0,  # neutral
    0.0,  # sadness
    0.0,  # surprise
]


top_10_books = recommend_books(user_input, book_emotions, df, top_n=10)

for idx, row in top_10_books.iterrows():
    print(f"\nTitle: {row['title']}")
    print(f"Author: {row['author']}")
    print(f"Rating: {row['rating']}")
    print(f"Description: {row['description']}")
    print(f"Language: {row['language']}")
    print(f"Genres: {row['genres']}")
    print(f"Pages: {row['pages']}")
    print(f"Publisher: {row['publisher']}")
    print(f"Publish Date: {row['publishDate']}")
    print(f"Number of Ratings: {row['numRatings']}")
    print(f"Liked Percent: {row['likedPercent']}")
    print(f"Cover Image URL: {row['coverImg']}")

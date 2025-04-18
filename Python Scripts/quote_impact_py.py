import pandas as pd
import numpy as np
import ast
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler
from sklearn.metrics import classification_report, roc_auc_score
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

tqdm.pandas()

# === STEP 1: Load and preprocess movie metadata === #
print("Loading metadata...")
metadata_df = pd.read_csv("./data/movie_titles_metadata.tsv", sep="\t", header=None,
                          names=["movie_id", "title", "year", "imdb_rating", "votes", "genres"])

# Safe parsing: if 'genres' is a string, parse it; otherwise, use an empty list
metadata_df["genres"] = metadata_df["genres"].apply(
    lambda x: ast.literal_eval(x.replace("'", '"')) if isinstance(x, str) else []
)

# === STEP 2: Load and preprocess movie quotes === #
print("Loading quotes...")
column_names = ['line_id', 'characterID', 'movie_id', 'characterName', 'quote']
quotes_df = pd.read_csv("./data/movie_lines.tsv", sep='\t', header=None, names=column_names, on_bad_lines='skip')
quotes_df = quotes_df.dropna(subset=['quote', 'movie_id'])[['movie_id', 'quote']]

# === STEP 3: Add 'impact' label manually for demo purposes === #
iconic_quotes = [
    "I'm going to make him an offer he can't refuse.",
    "May the Force be with you.",
    "You can't handle the truth!",
    "Here's looking at you, kid.",
    "I'll be back.",
    "Every man dies. Not every man really lives."
]

quotes_df["impact"] = quotes_df["quote"].apply(lambda x: 1 if x.strip().lower() in [q.lower() for q in iconic_quotes] else 0)

# === STEP 4: Merge quotes with metadata === #
merged_df = pd.merge(quotes_df, metadata_df, on="movie_id", how="left")
merged_df = merged_df.dropna(subset=['imdb_rating', 'votes'])

# === STEP 5: Feature engineering === #
print("Encoding genres and scaling metadata...")
mlb = MultiLabelBinarizer()
genre_encoded = mlb.fit_transform(merged_df["genres"])
genre_df = pd.DataFrame(genre_encoded, columns=mlb.classes_)

scaler = MinMaxScaler()
scaled_meta = scaler.fit_transform(merged_df[['imdb_rating', 'votes']])
meta_df = pd.DataFrame(scaled_meta, columns=['imdb_rating_scaled', 'votes_scaled'])

final_df = pd.concat([merged_df[['quote', 'impact']], genre_df, meta_df], axis=1)

# === STEP 6: Embed quotes === #
print("Embedding quotes...")
final_df['quote'] = final_df['quote'].fillna("").astype(str)
model = SentenceTransformer('all-MiniLM-L6-v2')
quote_embeddings = model.encode(final_df['quote'].tolist(), show_progress_bar=True)

# === STEP 7: Create final X, y datasets === #
X = np.hstack((quote_embeddings, final_df.drop(columns=['quote', 'impact']).values))
y = final_df['impact'].values

# === STEP 8: Train-test split === #
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === STEP 9: Train classifier === #
print("Training logistic regression...")
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

# === STEP 10: Evaluate === #
y_pred = clf.predict(X_test)
y_probs = clf.predict_proba(X_test)[:, 1]

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))
print("ROC AUC Score:", roc_auc_score(y_test, y_probs))

# === STEP 11: Try new quotes === #
test_quotes = [
    "Revenge is a dish best served cold.",
    "She went to the market to buy some bread.",
    "We are Groot.",
    "Why so serious?"
]

test_embeddings = model.encode(test_quotes)
test_metadata = np.zeros((len(test_quotes), X.shape[1] - quote_embeddings.shape[1]))  # use zeros if metadata not available
X_test_custom = np.hstack((test_embeddings, test_metadata))

scores = clf.predict_proba(X_test_custom)[:, 1]
print("\nQuote Impact Scores:")
for quote, score in zip(test_quotes, scores):
    print(f"\"{quote}\" → Impact Score: {score:.3f}")
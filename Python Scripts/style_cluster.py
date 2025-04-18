import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import seaborn as sns

# === STEP 1: Load and clean the dataset === #
# Define expected columns in the TSV file
column_names = ['line_id', 'characterID', 'movieID', 'characterName', 'text']

# Load data and handle bad lines
df = pd.read_csv(
    './data/movie_lines.tsv',
    sep='\t',
    names=column_names,
    encoding='utf-8',
    on_bad_lines='skip'
)

# Drop rows with missing dialogue text
df = df.dropna(subset=['text'])

# Keep only the text column for clustering
quotes = df['text'].astype(str).tolist()

# === STEP 2: Embed the quotes using Sentence-BERT === #
print("Loading embedding model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

print("Embedding quotes...")
embeddings = model.encode(quotes, show_progress_bar=True)

# === STEP 3: Apply KMeans Clustering === #
n_clusters = 5  # You can experiment with different values (e.g., 3 to 10)
print(f"Clustering into {n_clusters} groups...")
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(embeddings)

# Add cluster labels back to the original DataFrame
df['cluster'] = cluster_labels

# Save to CSV
df[['text', 'cluster']].to_csv('quote_clusters.csv', index=False)
print("Clustered quotes saved to quote_clusters.csv")

# === STEP 4: Visualize clusters using PCA === #
print("Reducing dimensions for visualization...")
pca = PCA(n_components=2)
reduced_embeddings = pca.fit_transform(embeddings)

df['x'] = reduced_embeddings[:, 0]
df['y'] = reduced_embeddings[:, 1]

plt.figure(figsize=(10, 7))
sns.scatterplot(data=df, x='x', y='y', hue='cluster', palette='tab10', alpha=0.7)
plt.title('Dialogue Style Clusters (PCA Projection)', fontsize=14)
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.legend(title='Cluster')
plt.tight_layout()
plt.savefig('cluster_plot.png')
plt.show()
print("Visualization saved to cluster_plot.png")
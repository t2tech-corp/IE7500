# Import Libraries
import ast
import re
import pandas as pd
import numpy as np
from tqdm import tqdm
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity


# Import the helper functions  
from dataset_helpers import *  
  
# Load the movie datasets  
movie_lines_df = movie_lines_read('data/movie_lines.tsv')  
movie_conversations_df = movie_conversations_read('data/movie_conversations.tsv')  
movie_characters_df = movie_characters_read('data/movie_characters_metadata.tsv')  
movie_titles_df = movie_titles_read('data/movie_titles_metadata.tsv')  

# Fix lineIDs parsing
def parse_line_ids(line_ids_str):
    try:
        # Remove any quotes and brackets, then split by spaces
        clean_str = line_ids_str.replace('[', '').replace(']', '').replace("'", "")
        return clean_str.split()
    except:
        return []

movie_conversations_df['lineIDs'] = movie_conversations_df['lineIDs'].apply(parse_line_ids)

# Fix genres parsing
def parse_genres(genres_str):
    try:
        # Remove any quotes and brackets, then split by spaces
        clean_str = genres_str.replace('[', '').replace(']', '').replace("'", "")
        return clean_str.split()
    except:
        return []

movie_titles_df['genres'] = movie_titles_df['genres'].apply(parse_genres)

# Create a dictionary to store all lines by their line_id, ensuring text is string
line_id_to_text = dict(zip(movie_lines_df['line_id'], movie_lines_df['text'].astype(str)))

# Function to get the text for a list of line IDs

def get_conversation_text(line_ids):
    conversation = []
    for line_id in line_ids:
        if line_id in line_id_to_text:
            # Convert to string in case it's not
            conversation.append(str(line_id_to_text[line_id]))
    return ' '.join(conversation)

# Create a new dataframe with movie_id and all its conversations
movie_conversations_list = []

for _, row in tqdm(movie_conversations_df.iterrows(), total=len(movie_conversations_df)):
    movie_id = row['movieID']
    conversation_text = get_conversation_text(row['lineIDs'])
    movie_conversations_list.append({
        'movieID': movie_id,
        'conversation_text': conversation_text
    })

conversations_df = pd.DataFrame(movie_conversations_list)

# Group by movie and concatenate all conversations
movie_texts = conversations_df.groupby('movieID')['conversation_text'].apply(' '.join).reset_index()

# Merge with movie titles to get genres
movie_texts = movie_texts.merge(movie_titles_df[['movieID', 'movieTitle', 'genres']], on='movieID', how='left')

# Fix the genres column by handling NaN values
def safe_genres(x):
    if isinstance(x, list):
        return x
    else:
        return []

movie_texts['genres'] = movie_texts['genres'].apply(safe_genres)

# Top 10 genres for analysis
top_genres = [g[0] for g in sorted_genres[:10]]

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Get English stopwords
stop_words = set(stopwords.words('english'))

# Add custom stopwords relevant to movie dialogues
custom_stopwords = {'oh', 'uh', 'um', 'yeah', 'hey', 'okay', 'ok', 'well', 'like', 'just', 'know', 'gonna', 'gotta', 'wanna'}

stop_words.update(custom_stopwords)

# Function to preprocess text
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords and lemmatize
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words and len(token) > 2]
    
    return ' '.join(tokens)

# Apply preprocessing to the conversation text
movie_texts['processed_text'] = movie_texts['conversation_text'].apply(preprocess_text)

# Use CountVectorizer on the processed text
vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')

# Use the processed_text column
text_data = movie_texts['processed_text'].tolist()
text_vectors = vectorizer.fit_transform(text_data)

# Define the number of topics
num_topics = 10

# Use sklearn's LDA
lda_model = LatentDirichletAllocation(n_components=num_topics, max_iter=10, learning_method='online', random_state=42)
lda_model.fit(text_vectors)

# Get topic distributions for each movie using LDA model transform
movie_topic_distributions = lda_model.transform(text_vectors)

# Add these distributions to a dataframe
topic_cols = [f'topic_{i}' for i in range(num_topics)]
movie_topics_df = pd.DataFrame(movie_topic_distributions, columns=topic_cols)
movie_topics_df['movieID'] = movie_texts['movieID'].values
movie_topics_df['genres'] = movie_texts['genres'].values
movie_topics_df['movieTitle'] = movie_texts['movieTitle'].values

# For simplicity, focus on the top 10 genres we identified earlier
top_genres = ['drama', 'thriller', 'action', 'comedy', 'crime', 'romance', 'sci-fi', 'adventure', 'mystery', 'horror']

# Create a dataframe to store average topic distribution for each genre
genre_topic_distribution = {genre: [] for genre in top_genres}

# For each movie, for each genre it belongs to (if in top_genres) assign its topic distribution
for idx, row in movie_topics_df.iterrows():
    for genre in row['genres']:
        if genre in top_genres:
            genre_topic_distribution[genre].append(row[topic_cols].values)

# Compute averages per genre
avg_genre_topic_dist = {}
for genre, distributions in genre_topic_distribution.items():
    if len(distributions) > 0:
        avg_genre_topic_dist[genre] = np.mean(distributions, axis=0)
    else:
        avg_genre_topic_dist[genre] = np.zeros(num_topics)

# Get the top 3 topics for each movie
def get_top_n_topics(topic_distribution, n=3):
    top_n_idx = topic_distribution.argsort()[-n:][::-1]
    return [(i, topic_distribution[i]) for i in top_n_idx]

# Create a feature table
feature_table = pd.DataFrame()
feature_table['movieID'] = movie_topics_df['movieID']
feature_table['movieTitle'] = movie_topics_df['movieTitle']
feature_table['genres'] = movie_topics_df['genres']

# Add top 3 topics and their scores
top_topics = []
for idx, row in movie_topics_df.iterrows():
    topic_dist = row[topic_cols].values
    top_topics.append(get_top_n_topics(topic_dist, n=3))
    
feature_table['top_topics'] = top_topics












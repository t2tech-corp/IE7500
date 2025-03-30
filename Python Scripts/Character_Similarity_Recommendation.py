# Libraries
import numpy as np  
import re  
from sentence_transformers import SentenceTransformer  
from sklearn.metrics.pairwise import cosine_similarity  

# Import the helper functions
from dataset_helpers import *

# Load the movie datasets
movie_lines = movie_lines_read('data/movie_lines.tsv')
movie_conversations = movie_conversations_read('data/movie_conversations.tsv')
movie_characters = movie_characters_read('data/movie_characters_metadata.tsv')
movie_titles = movie_titles_read('data/movie_titles_metadata.tsv')

# Create Movie Dialogue
movie_dialogue_df = movie_lines.groupby(['movieID', 'characterName'], as_index=False)['text'].apply(lambda texts: ' '.join([str(t) for t in texts if pd.notna(t)])).rename(columns={'text': 'movie_dialogue'})

# Text Preprocessing function: lowercasing, whitespace removal  
def preprocess_text(text):  
    text = text.lower()  
    text = re.sub(r'\s+', ' ', text)  
    return text.strip() 

# Preprocess the aggregated dialogue in movie_dialogue_df  
movie_dialogue_df['clean_dialogue'] = movie_dialogue_df['movie_dialogue'].apply(preprocess_text)

# Load the pre-trained SentenceTransformer model  
model_name = 'all-MiniLM-L6-v2'  # A lightweight model  
model = SentenceTransformer(model_name)  

# Generate embeddings from the cleaned dialogues  
dialogues_list = movie_dialogue_df['clean_dialogue'].tolist()  
embeddings = model.encode(dialogues_list, show_progress_bar=True)  
embeddings = np.array(embeddings)  

# Compute the cosine similarity between dialogues  
similarity_matrix = cosine_similarity(embeddings) 

# Function to retrieve top similar entries (excluding self)  
def get_top_similar(sim_matrix, index, top_n=3):  
    sim_scores = list(enumerate(sim_matrix[index]))  
    sim_scores[index] = (index, -1)  # exclude self  
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)  
    return sim_scores[:top_n]

# For each character, get the top 3 most similar characters 
# Only accept recommendations if characters are from different movies
recommendations = []  

for i in range(len(movie_dialogue_df)):  
    top_sim = get_top_similar(similarity_matrix, i, top_n=10)  
    count = 0 
    for rank, (j, score) in enumerate(top_sim, start=1):  
        if movie_dialogue_df.iloc[i]['movieID'] != movie_dialogue_df.iloc[j]['movieID']:  
            count += 1  
            recommendations.append({  
                'source_movieID': movie_dialogue_df.iloc[i]['movieID'],  
                'source_characterName': movie_dialogue_df.iloc[i]['characterName'],  
                'target_movieID': movie_dialogue_df.iloc[j]['movieID'],  
                'target_characterName': movie_dialogue_df.iloc[j]['characterName'],  
                'similarity': score,  
                'rank': rank  
            })  

        # Only take the top 3 recommendations per source character  
        if count >= 3:  
            break 
  
recommendations_df = pd.DataFrame(recommendations)  

# Merge in movie titles
movie_id_to_title = dict(zip(movie_titles['movieID'], movie_titles['movieTitle']))  

# Convert movie titles to uppercase  
movie_id_to_title = {k: v.upper() for k, v in movie_id_to_title.items()} 

recommendations_df['source_movieTitle'] = recommendations_df['source_movieID'].map(movie_id_to_title)  
recommendations_df['target_movieTitle'] = recommendations_df['target_movieID'].map(movie_id_to_title)  
 
# Reorder columns for clarity  
recommendations_df = recommendations_df[['source_movieID', 'source_movieTitle', 'source_characterName',  
                                        'target_movieID', 'target_movieTitle', 'target_characterName',  
                                        'similarity', 'rank']] 

# Optional to Save to CSV
recommendations_df.to_csv('recommendations.csv', index=False)
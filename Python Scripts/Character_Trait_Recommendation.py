# Import necessary libraries  
import pandas as pd  
import numpy as np  
import torch  
from transformers import pipeline  
from tqdm.notebook import tqdm  
import matplotlib.pyplot as plt  
import seaborn as sns  
from sklearn.metrics.pairwise import cosine_similarity  

# Import the helper functions  
from dataset_helpers import *  
  
# Load the movie datasets  
movie_lines = movie_lines_read('data/movie_lines.tsv')  
movie_conversations = movie_conversations_read('data/movie_conversations.tsv')  
movie_characters = movie_characters_read('data/movie_characters_metadata.tsv')  
movie_titles = movie_titles_read('data/movie_titles_metadata.tsv')  

# Create Movie Dialogue  
movie_dialogue_df = movie_lines.groupby(['movieID', 'characterName'], as_index=False)['text'].apply(  
    lambda texts: ' '.join([str(t) for t in texts if pd.notna(t)])  
).rename(columns={'text': 'movie_dialogue'})  

# Create a mapping from movie ID to title for later use  
movie_id_to_title = dict(zip(movie_titles['movieID'], movie_titles['movieTitle']))  

# Define character traits to analyze  
traits = [  
    "friendly", "hostile",  
    "introverted", "extroverted",  
    "intelligent", "naive",  
    "courageous", "cowardly",  
    "compassionate", "ruthless",  
    "funny", "serious",  
    "honest", "deceptive",  
    "loyal", "selfish"  
]  

# Load the zero-shot classification pipeline  
classifier = pipeline(  
    "zero-shot-classification",   
    model="roberta-large-mnli",   
    device=device  
)  

# Function to analyze traits for a given text  
def analyze_traits(text, traits_list, classifier, min_words=10):  
    # Skip very short texts  
    if len(str(text).split()) < min_words:  
        return {trait: 0.0 for trait in traits_list}  
      
    # Truncate very long texts (to avoid CUDA memory issues)  
    max_length = 1024  
    words = str(text).split()  
    if len(words) > max_length:  
        text = ' '.join(words[:max_length])  
      
    try:  
        # Run zero-shot classification  
        result = classifier(  
            text,   
            candidate_labels=traits_list,  
            multi_label=True  
        )  
          
        # Extract scores  
        scores = {label: score for label, score in zip(result['labels'], result['scores'])}  
        return scores  
    except Exception as e:  
        print(f"Error analyzing text: {e}")  
        return {trait: 0.0 for trait in traits_list} 

# Process characters in batches to avoid memory issues  
batch_size = 100  
all_results = []  
  
print(f"Analyzing traits for {len(movie_dialogue_df)} characters...")  
for i in range(0, len(movie_dialogue_df), batch_size):  
    batch = movie_dialogue_df.iloc[i:i+batch_size]  
    print(f"Processing batch {i//batch_size + 1}/{(len(movie_dialogue_df)-1)//batch_size + 1}...")  
      
    for _, row in tqdm(batch.iterrows(), total=len(batch)):  
        movie_id = row['movieID']  
        character_name = row['characterName']  
        dialogue = row['movie_dialogue']  
          
        # Analyze traits  
        trait_scores = analyze_traits(dialogue, traits, classifier)  
          
        # Add to results  
        result = {  
            'movieID': movie_id,  
            'movieTitle': movie_id_to_title.get(movie_id, 'Unknown'),  
            'characterName': character_name  
        }  
        result.update(trait_scores)  
        all_results.append(result)  
      
    # Save intermediate results  
    temp_df = pd.DataFrame(all_results)  
    temp_df.to_csv('trait_analysis_intermediate.csv', index=False)  
    print(f"Saved intermediate results ({len(all_results)} characters processed)")  

# Create final trait dataframe  
trait_df = pd.DataFrame(all_results)  

# Compute similarity matrix    
trait_vectors = trait_df[traits].values  
similarity_matrix = cosine_similarity(trait_vectors)  

# Generate recommendations
recommendations = [] 

# Get indices for each character  
indices = list(range(len(trait_df)))  

# For each character  
for i in tqdm(indices):  
    source_movie_id = trait_df.iloc[i]['movieID']  
    source_character = trait_df.iloc[i]['characterName']  
      
    # Find similar characters from different movies  
    similar_indices = []  
    for j in indices:  
        if i != j and trait_df.iloc[j]['movieID'] != source_movie_id:  # Ensure different movie  
            similar_indices.append((j, similarity_matrix[i, j]))  
      
    # Sort by similarity  
    similar_indices.sort(key=lambda x: x[1], reverse=True)  
      
    # Get top recommendations  
    top_n = 10  
    for rank, (idx, similarity) in enumerate(similar_indices[:top_n], 1):  
        if similarity > 0.5:  # Only include if similarity is above threshold  
            target_movie_id = trait_df.iloc[idx]['movieID']  
            target_character = trait_df.iloc[idx]['characterName']  
              
            recommendations.append({  
                'source_movieID': source_movie_id,  
                'source_movieTitle': movie_id_to_title.get(source_movie_id, 'Unknown'),  
                'source_characterName': source_character,  
                'target_movieID': target_movie_id,  
                'target_movieTitle': movie_id_to_title.get(target_movie_id, 'Unknown'),  
                'target_characterName': target_character,  
                'similarity': similarity,  
                'rank': rank  
            })  

# Create recommendations dataframe  
recommendations_df = pd.DataFrame(recommendations)  

# Sort and select columns  
recommendations_df = recommendations_df.sort_values(['source_movieID', 'source_characterName', 'rank'])  
recommendations_df = recommendations_df[[  
    'source_movieID', 'source_movieTitle', 'source_characterName',   
    'target_movieID', 'target_movieTitle', 'target_characterName',   
    'similarity', 'rank'  
]]  

# Convert movie titles to uppercase 
# Convert movie titles in recommendations_df to uppercase  
recommendations_df['source_movieTitle'] = recommendations_df['source_movieTitle'].str.upper()  
recommendations_df['target_movieTitle'] = recommendations_df['target_movieTitle'].str.upper()

# Optional Save recommendations  
recommendations_df.to_csv('character_recommendations_roberta.csv', index=False) 
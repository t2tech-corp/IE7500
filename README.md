# IE7500 
Github Repo for IE7500 Course Project | Cinematic Quote Analyzer

This repository contains the code and resources for our semester project on Natural Language Processing (NLP).

## Table of Contents

- [Introduction](#introduction)
- [Project Description](#project-description)
- [Literature Review](#literature-review)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Implementation](#implementation)
- [Requirements](#requirements)
- [Usage](#usage)
- [Results](#results)
- [Future Work](#future-work)
- [License](#license)
- [Team Members](#team-members)

## Introduction

Iconic movie quotes have the power to evoke emotions, spark conversations, and create lasting impressions on audiences. 
These quotes often encapsulate a film’s emotional core, character depth, or overarching themes. Despite their cultural significance,
their broader emotional and thematic contexts are rarely analyzed or used as a gateway to discover related films. This gap leaves
untapped potential for audiences who wish to explore the connections between their favorite quotes and the cinematic universe they belong to.

## Project Description

With our Cinematic Quote Analyzer, we aim to develop an interactive tool that empowers users to explore the deeper meaning behind their favorite movie quotes.
Users will be able to input any cinematic quote, which the tool will analyze for its emotional tone, genre association, and contextual relevance within the script.
Leveraging advanced natural language processing techniques like sentiment analysis, topic modeling, and contextual embeddings, the tool will provide meaningful 
insights into the quote's significance. We also plan for the tool to offer tailored movie recommendations based on themes, tones, or dialogue styles, fostering a 
deeper appreciation for cinematic storytelling and connecting users with similar films they might enjoy!

## Literature Review

The application of Natural Language Processing (NLP) to movie dialogue and character datasets has opened up exciting avenues for research, particularly in areas like quote similarity, character trait analysis, and genre matching.

**Character Trait Analysis**
* Character Embeddings

   Researchers are developing methods to create "character embeddings" that capture the essence of a character's personality through their dialogue. This involves training models to understand not just the words spoken, but also the context in which they are spoken, and the interactions between characters.   
   Works like "Representing Movie Characters in Dialogues" by Azab et al. (2019) explore how to encode both the language used by characters and their interactions into a single representation. This allows for more accurate character comparisons.   
   The ability to understand character traits through dialogue enables the development of tools that can analyze character development, identify archetypes, and even predict character behavior.   

* Sentiment Analysis

   NLP techniques are used to analyze the sentiment expressed in dialogue, providing insights into a character's emotional state. This can be used to track emotional arcs and identify key moments of emotional change.   
   Analyzing the frequency and intensity of certain words or phrases can reveal underlying personality traits.   

**Quote Similarity**
* Semantic Similarity

   NLP techniques like word embeddings (e.g., Word2Vec, GloVe, BERT) and sentence embeddings are used to determine the semantic similarity between quotes.   
   This allows for the identification of quotes that have similar meanings, even if they use different words.   
   This is valuable for tasks like identifying recurring themes or motifs in a film, or comparing dialogue across different films.   

**Genre Matching**
* Dialogue Features

   NLP techniques are used to extract features from dialogue that are indicative of specific genres. For example, action movies may feature more short, concise sentences, while romantic comedies may feature more emotional and expressive dialogue.   
   Researchers analyze factors like vocabulary, sentiment, and dialogue structure to identify genre-specific patterns.   

* Topic Modeling

   Topic modeling techniques (e.g., LDA) are used to identify the main topics discussed in a film's dialogue. This can help to classify films into genres based on their thematic content.   
   "Evaluating Cinematic Dialogue - Which syntactic and semantic features are predictive of genre? | Towards Data Science" provides good information on this topic.   

## Dataset

Cornell University Movie Dialogue Corpus
This corpus comes from the paper, "Chameleons in imagined conversations: A new approach to understanding coordination of linguistic style in dialogs" by Cristian Danescu-Niculescu-Mizil and Lillian Lee.

The paper and up-to-date data can be found here: http://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html

This corpus contains a metadata-rich collection of fictional conversations extracted from raw movie scripts:
* movie_titles_metadata.txt (41.97 kB)
* movie_characters_metadata.txt (353.34 kB)
* movie_lines.txt (25.04 MB)
* movie_conversations.txt (4.63 MB)
* raw_script_urls.txt (46.89 kB)

Source of Datasets
* https://www.kaggle.com/datasets/Cornell-University/movie-dialog-corpus?resource=download

Size of Datasets
* 220,579 conversational exchanges between 10,292 pairs of movie characters.
* 9,035 characters from 617 movies
* 304,713 utterances

Key Features of the Datasets
* movie_titles_metadata.txt
	- contains information about each movie title
	- fields: 
		- movieID, 
		- movie title,
		- movie year,
	        - IMDB rating,
		- no. IMDB votes,
 		- genres in the format ['genre1','genre2','genreN']
* movie_characters_metadata.txt
	- contains information about each movie character
	- fields:
		- characterID
		- character name
		- movieID
		- movie title
		- gender ("?" for unlabeled cases)
		- position in credits ("?" for unlabeled cases)
* movie_lines.txt
	- contains the actual text of each utterance
	- fields:
		- lineID
		- characterID (who uttered this phrase)
		- movieID
		- character name
		- text of the utterance
* movie_conversations.txt
	- the structure of the conversations
	- fields
		- characterID of the first character involved in the conversation
		- characterID of the second character involved in the conversation
		- movieID of the movie in which the conversation occurred
		- list of the utterances that make the conversation, in chronological 
			order: ['lineID1','lineID2','lineIDN']
			has to be matched with movie_lines.txt to reconstruct the actual content
* raw_script_urls.txt
	- the urls from which the raw sources were retrieved

## Methodology

Explanation of the multiple NLP techniques and models tested for the project.

**Sentiment Analysis**

TextBlob for Movie Lines
* Preprocessing steps: None
* Feature extraction methods: Selected "text" field (movie line) to analyze.
* Models used: TextBlob from textblob
* Evaluation metrics: sentiment.polarity

TextBlob for Movie Conversations
* Preprocessing steps: Create full conversation from movie lines and conversations.
* Feature extraction methods: Selected "text" field (movie conversations) to analyze.
* Models used: TextBlob from textblob
* Evaluation metrics: sentiment.polarity

**Classification**

BERT for Movie Lines and Movies Metadata
* Preprocessing steps: Merged dataset of movie lines and movie metadata.
* Feature extraction methods: Movie Line and Classification extraction. Autotokenizer from transformers.
* Models used: bert-base-uncased from Torch. AutoModelForSequenceClassification from transformers.
* Evaluation metrics: Confusion matrix for predicted versus true predictions.

Huggingface for Character Trait Classification
* Preprocessing steps: Merged dataset of movie lines by character, movie, dialogue.
* Feature extraction methods: nltk stopwords, punkt, wordnet, lemmatization.
* Models used: zero-shot-classification, facebook/bart-large-mnli.
* Evaluation metrics: Dominant Trait Characteristic.

**Similarity**

Facebook AI Similarity Search (FAISS) for Character Similarity
* Preprocessing steps: Merged dataset of movie lines by character, movie, dialogue.
* Feature extraction methods: nltk stopwords, punkt, punkt_tab, wordnet, lemmatization, embeddings.
* Models used: faiss.
* Evaluation metrics: Candidate indices similarity scores.

## Implementation

* Programming language used: Python.
* Libraries used: NLTK, spaCy, scikit-learn, TensorFlow, PyTorch, SentenceTransformer, faiss, TextBlob, transformers.
* Structure of the code: All source datasets are located in Movie Dialogue Corpus sub-folder. Code is in Python with commented sections to describe functionality. Output, visualizations, and other information can be found in Jupyter Notebooks.

## Requirements

All code can be run using the provided Python scripts and notebooks with the related datasets found in the corpus. Final Python and library versions will be provided upon final model decisions.

## Usage

Proper usage of the finalized scripts and selected models will be provided upon completion of project.

## Results

Results for models selected will be provided upon project completion once model refinements are settled. Results criteria will be as follows:

* Performance metrics on the evaluation dataset.
* Comparisons to other approaches (if applicable).
* Any visualizations or graphs to illustrate the results.

## Future Work
[Discuss potential improvements or extensions to the project.  For example:

Exploring different models or techniques.
Expanding the dataset.
Improving the performance.
Adding new features.]

## License
The MIT License (MIT)

Copyright (c) 2015 Chris Kibble

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## Team Members
- Adrian Rodriguez 
- Priska Mohunsingh 
- Tim Terry 



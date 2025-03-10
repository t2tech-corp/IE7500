# IE7500 
Github Repo for IE7500 Course Project | Cinematic Quote Analyzer

This repository contains the code and resources for our semester project on Natural Language Processing (NLP).

## Table of Contents

- [Introduction](#introduction)
- [Project Description](#project-description)
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

## Datasets

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

[Explain the NLP techniques and models used in your project.  For example:
* Preprocessing steps (tokenization, stemming, lemmatization, etc.).
* Feature extraction methods (TF-IDF, word embeddings, etc.).
* Models used (Naive Bayes, Logistic Regression, RNNs, Transformers, etc.).
* Evaluation metrics (accuracy, precision, recall, F1-score, etc.).]

## Implementation

[Provide details about the implementation:
* Programming languages used (e.g., Python).
* Libraries used (e.g., NLTK, spaCy, scikit-learn, TensorFlow, PyTorch).
* Structure of the code (briefly describe the different modules/files).  If the structure isn't obvious from the file names, explain the purpose of important files.
* How to run the code (see the Usage section below).]

## Requirements

[List the software and libraries required to run the project.  It's best to include a `requirements.txt` file for easy installation:]

## Usage
[Provide clear instructions on how to run the code.  Include specific commands and examples.Include any necessary configuration steps or data preparation steps.]

## Results
[Summarize the results of your project.  Include:

Performance metrics on the evaluation dataset.
Comparisons to other approaches (if applicable).
Any visualizations or graphs to illustrate the results.]

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



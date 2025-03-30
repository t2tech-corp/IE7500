# Helper functions for reading Movie datasets using standard functions
# Within scripts, add the following to use: 

# from dataset_helpers import *

# Sample Usage to read Movie Lines
#
# fileloc = 'data/movie_lines.tsv'
#
# movie_lines = movie_lines_read(fileloc)
#
# print(movie_lines.head())

import pandas as pd

# Function to read Movie datasets using file location (fileloc) and column names (colnames)
def read_file(fileloc, colnames):

    df = pd.read_csv(fileloc, sep='\t', encoding='utf-8', header=None, names=colnames, on_bad_lines='skip', quoting=3)

    return df


# Movie Lines Dataset Creation
def movie_lines_read(fileloc):

    # Define column names
    column_names = ['line_id', 'characterID', 'movieID', 'characterName', 'text']

    # Call read_file function with file location
    df = read_file(fileloc, column_names)

    return df


# Movie Conversations Dataset Creation
def movie_conversations_read(fileloc):

    # Define column names
    column_names = ['characterID_1', 'characterID_2', 'movieID', 'lineIDs']

    # Call read_file function with file location
    df = read_file(fileloc, column_names)

    return df


# Movie Characters Dataset Creation
def movie_characters_read(fileloc):

    # Define column names
    column_names = ['characterID', 'characterName', 'movieID', 'movieTitle', 'gender', 'creditsPos']

    # Call read_file function with file location
    df = read_file(fileloc, column_names)

    return df


# Movie Titles Dataset Creation
def movie_titles_read(fileloc):

    # Define column names
    column_names = ['movieID', 'movieTitle', 'movieYear', 'IMDBRating', 'IMBDVotes', 'genres']

    # Call read_file function with file location
    df = read_file(fileloc, column_names)

    return df
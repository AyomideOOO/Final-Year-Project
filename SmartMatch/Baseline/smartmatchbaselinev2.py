# Baseline v2
# Author: Ayomide Osineye

# Pandas used for storing csvs as dataframes: df, pf 
import pandas as pd


# K - integer
# numpy used to get the Top K recommended roles 
import numpy as np


# Implemented TF-IDF for vectorizing the user input
# and the 'relevant fields' dataframe.
from sklearn.feature_extraction.text import TfidfVectorizer


# Implemented to calculate the similarity scores between the 
# relevant fields and the user input (vectorized)

from sklearn.metrics.pairwise import cosine_similarity

# matplotlib used for plotting graphs to visualise similarity scores and role names
import matplotlib.pyplot as plt


class SmartMatchBaseline:

    # Constructor: initializes the SmartMatchBaseline object
    # df: the original dataset
    # pf: DataFrame containing combined relevant fields from df
    # X: TF-IDF vectors of the relevant fields
    # Y: TF-IDF vector of the user query

    def __init__(self):
        self.df = None
        self.X = None
        self.Y = None
        self.pf = None

    
    # 'readcsv' reads the provided csv file. 
    #  If the size of the file is greater than 75000, Dataframe 'df' will contain the rows.
    #  If not the whole csv will be read into 'df'

    def readcsv (self, my_csv ):
        if len(pd.read_csv(my_csv)) > 75000:
            self.df = pd.read_csv(my_csv).head(75000)
        else:
            self.df = pd.read_csv(my_csv)


    def combine_relevant_fields (self, relevant_columns, my_csv):

        # Create a new DataFrame 'pf' with one column 'Relevant_Fields'
        self.pf = pd.DataFrame(columns= ['Relevant_Fields'])

       
        # Data cleaning: fillna stores NaN values as empty strings.
        # this is to prevent data loss from concatinating rows. 

        # Store the first relevant column from 'df' into 'pf', converting text to lowercase
        self.pf['Relevant_Fields'] = self.df[relevant_columns[0]].fillna("").str.lower()

        
        # For loop stores the remaining relevant columns from 'df' into 'pf' 
        # concatenating each column with a comma.

        for i in range(1,len(relevant_columns)):
            self.pf['Relevant_Fields'] += ", " + self.df[relevant_columns[i]].fillna("").str.lower() 


        # Stores new Dataframe 'pf' as a CSV file.
        self.pf.to_csv(my_csv, index= False)


    def convert_to_vectors(self):

        # vectorizes the column 'Relevant_Fields' in dataframe 'pf' using TF-IDF
        vectorizer = TfidfVectorizer()
        self.X = vectorizer.fit_transform(self.pf['Relevant_Fields'])

        # ask the user for what placement they are looking for 
        user_input = input("Enter the type of role you are looking for: ")

        # vectorize the user input
        self.Y = vectorizer.transform([user_input])

        return self.X, self.Y
    

    # NEED TO REWRITE COMMENT HERE ON A LATER DAY


    def similarity_score(self):
       X, Y = self.convert_to_vectors()
       similarity_score = cosine_similarity(X,Y)
    
       return similarity_score
    

    def top_recommendations(self,k):

        # flatten converts multi-dimensional array into a 1-D array
        similarity_score = self.similarity_score().flatten()

        # argsort sorts the indexes of highest values to lowest values in asending order (highest -> list[len(list-1)]) 
        # (lowest - list[0])

        k_best = np.argsort(similarity_score)[-k:][::-1]

        n = len(k_best)

        for i in range(n):
            index = k_best[i]
            print(f" {i+1}) {self.df['title'][index]} \n\n {self.df['description'][index]} \n\n")
        
    
    # next session: Use hash-map to combine indexes with their respective cosine_similarity values
    # make a graph and plot the x axis -> the cosine similarities + role names, y axis -> 0 -to 1. 
    # this would show the relationship in the similarity and why certain roles showed in top k recommendations

    def graph_representation(self,k):

        similarity_score = self.similarity_score().flatten()
        best = np.sort(similarity_score)[-k:][::-1]

        print(best)
      


# New instance of baseline model.
real_dataset = SmartMatchBaseline()
real_dataset.readcsv('postings.csv')


# get all the relevant fields together into new dataframe
relevant_columns= ['title', 'location', 'skills_desc', 'description']
real_dataset.combine_relevant_fields(relevant_columns, 'relevant_fields.csv')


# TF-IDF and Cosine Similarity + Top K recommendations
real_dataset.top_recommendations(3)


# byte-encoder scoring
# text encoders for context - BERT, berta




# Things to do later (written 8/12/20205) 
# - Measure execution time of the algorithm
# - Add byte-encoder scoring for semantic reasoning (BERT, berta etc.)
# - Experiment with dataset size for optimal performance (quote in report)
# - Randomize dataset before sampling top 300 rows if >75,000 entries
# - For data cleaning, drop roles with NaN values instead of filling the spaces (.fillna)

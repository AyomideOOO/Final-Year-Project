# Baseline v2
# Author: Ayomide Osineye
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from heapq import nlargest
import matplotlib.pyplot as plt
# import time


class SmartMatchBaseline:


    # constructor 
    def __init__(self):
        self.df = None
        self.X = None
        self.Y = None
        self.pf = None


    def readcsv (self, my_csv ):

        if len(pd.read_csv(my_csv)) > 75000:

            # if the size is bigger than 300 then randomize the dataset for 
            # potentially better results then get the first 300 results. (IMPLEMENT LATER POTENTIALLY)
            # self.df = pd.read_csv(my_csv).sample(n=300).reset_index(drop=True)

            self.df = pd.read_csv(my_csv).head(75000)
        else:
            self.df = pd.read_csv(my_csv)



    def combine_relevant_fields (self, relevant_columns, my_csv):
        self.pf = pd.DataFrame(columns= ['Relevant_Fields'])

        # combine all the fields in relevant_columns into relevant_fields
        # for each role.

        # fillna used to deal with NaN values. When combining string values, if a column
        # includes a NaN value the whole string will become NaN.
        # data cleaning.

        # drop rows with NAN values

        self.pf['Relevant_Fields'] = self.df[relevant_columns[0]].fillna("").str.lower()

        for i in range(1,len(relevant_columns)):
            self.pf['Relevant_Fields'] += ", " + self.df[relevant_columns[i]].fillna("").str.lower() 
        self.pf.to_csv(my_csv, index= False)


    def convert_to_vectors(self):

        # convert the the relevant fields to vector values using TF-IDF
        vectorizer = TfidfVectorizer()
        self.X = vectorizer.fit_transform(self.pf['Relevant_Fields'])

        # ask the user for what placement they are looking for 
        user_input = input("Enter the type of role you are looking for: ")

        # vectorize the user input
        self.Y = vectorizer.transform([user_input])

        return self.X, self.Y
    

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
      

# Testing out class
# new = SmartMatchBaseline()
# new.readcsv("baseline_dataset.csv")


# get the relevant columns = 
# relevant = ['Job_title', 'Location', 'Skills', 'Industry']
# new.combine_relevant_fields(relevant, 'new_test.csv')


# top recommendations
# new.top_recommendations(2)

# see whats a good dataset size and quote it.


# New instance of baseline model.
real_dataset = SmartMatchBaseline()
real_dataset.readcsv('postings.csv')


# get all the relevant fields together into new dataframe
relevant = ['title', 'location', 'skills_desc', 'description']
real_dataset.combine_relevant_fields(relevant, 'relevant_fields.csv')


# TF-IDF and Cosine Similarity + Top K recommendations
real_dataset.top_recommendations(3)




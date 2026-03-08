# Baseline v2
# Author: Ayomide Osineye


# K - integer
# numpy used to get the Top K recommended roles 
import numpy as np


# Implemented to calculate the similarity scores between the 
# relevant fields and the user input (vectorized)
from sklearn.metrics.pairwise import cosine_similarity

# matplotlib used for plotting graphs to visualise similarity scores and role names
import matplotlib.pyplot as plt

# time used for working out run-time
import time 

# streamlit used for the user interface
import streamlit as st 


class Baseline:
    # Using composition to access methods / variables unique to preprocessing class
    def __init__(self, preprocessor):
        self.preprocessor = preprocessor



    # returns the similarity scores between job postings and User CV

    def similarity_score(self, X, Y):
       similarity_score = cosine_similarity(X,Y)
       return similarity_score
    

    def top_recommendations(self,k, similarity_scores):

        # flatten converts multi-dimensional array into a 1-D array
        scores = similarity_scores.flatten()

        # argsort sorts the indexes of highest values to lowest values in asending order (highest -> list[len(list-1)]) 
        # (lowest -> list[0])

        k_best = np.argsort(scores)[-k:][::-1]
        n = len(k_best)

        for i in range(n):
            index = k_best[i]
            st.text(f" {i+1}) {self.preprocessor.df['title'][index]} \nSimilarity Score: {scores[index]} \n\n {self.preprocessor.df['description'][index]} \n\n",)
    

    # next session: Use hash-map to combine indexes with their respective cosine_similarity values
    # make a graph and plot the x axis -> the cosine similarities + role names, y axis -> 0 -to 1. 
    # this would show the relationship in the similarity and why certain roles showed in top k recommendations

    def graph_representation(self,k):
        similarity_score = self.similarity_score().flatten()
        best = np.sort(similarity_score)[-k:][::-1]
        print(best)




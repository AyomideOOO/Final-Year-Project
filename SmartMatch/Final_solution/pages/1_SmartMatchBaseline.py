# Baseline v2
# Author: Ayomide Osineye



import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

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
from preprocessing import Preprocessing



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

        # argsort sorts the indexes of highest values to lowest values in asending order (highest -> list[-1]) 
        # (lowest -> list[0])
        # argsort returns indexes that will sort the array in ascending order where the highest score is at the end  

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


# Running Baseline Page

# Caches preprocessing (data loading, feature extraction, TF-IDF fitting) to
# avoid recomputation, reducing execution time. 
@st.cache_resource
def start_program():
    dataset = Preprocessing()
    st.title("Baseline")

    dataset.readcsv("SmartMatch/Data/postings.csv")
    relevant_columns = ['title', 'location', 'skills_desc', 'description']
    dataset.combine_relevant_fields(relevant_columns)

    X = dataset.convert_postings_TFIDF()
    return X, dataset
    

def processing(X, dataset, k):
    # instance of baseline model created
    baseline = Baseline(dataset)

    # Store cv data in user_cv
    user_cv = dataset.upload_file()

    # Processing can continue if the user uploads their CV. 
    if user_cv is not None:

        # Vectorize the user cv using TF-IDF 
        Y = dataset.convert_user_TFIDF(user_cv)

        # Similarity scores between user query and job postings computed using cosine similarity
        Similarity_Scores = baseline.similarity_score(X, Y)
        
        # Top K job recommendations retrieved and outputted to user. 
        baseline.top_recommendations(k,Similarity_Scores)
        


## Running First Page
start = time.time()
X, dataset = start_program()
end = time.time()

total_time = end - start
print("Page Load/Reload :",total_time)
processing(X, dataset, 3)



# Smart Match Page


# NOTE: Implemented to help with 'ModuleNotFoundError' due my file structure
#       Used compositon to integrate the preprocessing class but because it is higher
#       in the hierarchy of files, it cannot be seen 
import sys
from pathlib import Path

# NOTE: parent.parent = pages (folder) -> Final_solution (folder)
#       __file__ = current file (2_SmartMatch.py)
#       sys.path.append(str(  )) = adds path containing preprocessing.py so Preprocessing can be found  
sys.path.append(str(Path(__file__).resolve().parent.parent))

from preprocessing import Preprocessing
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import streamlit as st
import numpy as np
import time

class smartmatch:

    # composition
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
            st.text(f" {i+1}) {self.preprocessor.df['title'][index]} ,\nSimilarity Score: {scores[index]} \n\n {self.preprocessor.df['description'][index]} \n\n",)
        

    
# Running SmartMatch Page

# Caches preprocessing (data loading, feature extraction, TF-IDF fitting) to
# avoid recomputation, reducing execution time. 
@st.cache_resource
def start_program():
    dataset = Preprocessing()
    st.title("SmartMatch System")

    dataset.readcsv("SmartMatch/Data/postings.csv")
    relevant_columns = ['title', 'location', 'skills_desc', 'description']
    dataset.combine_relevant_fields(relevant_columns)
    X = dataset.convert_postings_embeddings()


    return X, dataset


def processing(X, dataset, k):
        
        # instance of smartmatch class 
        running = smartmatch(dataset)

        # Store cv data in user_cv
        user_cv = dataset.upload_file()

        # Processing can continue if the user uploads their CV. 
        if user_cv is not None:

            # Vectorize the user cv using TF-IDF
            Y = dataset.convert_user_embeddings(user_cv)

            # Similarity scores between user query and job postings computed using cosine similarity
            Similarity_Scores = running.similarity_score(X, Y)
            
            # Top K job recommendations retrieved and outputted to user. 
            running.top_recommendations(k,Similarity_Scores)


    
start = time.time()
X, dataset = start_program()
end = time.time()

total_time = end - start
print("Page Load/Reload :",total_time)
processing(X,dataset, 3)

    

    


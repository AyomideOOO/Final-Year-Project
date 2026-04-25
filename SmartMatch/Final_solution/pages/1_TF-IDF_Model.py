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

# time used for working out run-time
import time 

# streamlit used for the user interface
import streamlit as st 
from preprocessing import Preprocessing



class Baseline:
    # Using aggregation to access methods / variables unique to preprocessing class
    def __init__(self, preprocessor):
        self.preprocessor = preprocessor


    # returns the similarity scores between job postings and User CV
    def similarity_score(self, X, Y):
       similarity_score = cosine_similarity(X,Y)
       return similarity_score
    

    def top_recommendations(self,k, similarity_scores):

        # flatten converts multi-dimensional array into a 1-D array
        scores = similarity_scores.flatten()

        # argsort returns indices that sort scores in ascending order (lowest → highest)
        # [-k:] slices the last K indices (highest scoring)
        # [::-1] reverses to descending order
        k_best = np.argsort(scores)[-k:][::-1]
        n = len(k_best)

        for i in range(n):
            index = k_best[i]

            # returns the relevant information based on each role returned: Job title, Similarity Score, and description
            title = self.preprocessor.df['title'][index]
            score = scores[index]
            desc = self.preprocessor.df['description'][index]

            # Short preview (first 200 characters)
            short_desc = desc[:200] + "..."

            st.write(f"**{i+1}) {title}**")

            st.write(f"Similarity Score: {score:.4f}")
            st.write(short_desc)

            # Expandable full description
            with st.expander("See full description"):
                st.write(desc)

            st.write("-" * 50)
            

# Running TF-IDF Page
st.set_page_config(page_title="TF-IDF Baseline")
st.title("TF-IDF Model")

# Caches preprocessing (data loading, feature extraction, TF-IDF fitting) to
# avoid recomputation, reducing execution time. 
@st.cache_resource
def start_program():

    dataset = Preprocessing()
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
        


# Runtime computation
# time taken at different components of the pipeline
start = time.time()
X, dataset = start_program()
end_precomputation = time.time()

processing(X,dataset, 5)
end_computation = time.time()
 

print("Preprocessing time: :", end_precomputation - start)
print("Retrieval time:", end_computation - end_precomputation)
print("Total runtime: ", end_computation - start)
print("-" * 50)




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
import streamlit as st
import numpy as np
import time


class SmartMatch:

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

    
# Running Embeddings Page
st.set_page_config(page_title="Embeddings Model")
st.title("Sentence Embeddings Model")
st.subheader("Model: all-MiniLM-L6-v2")



# Caches preprocessing (data loading, feature extraction, TF-IDF fitting) to
# avoid recomputation, reducing execution time. 
@st.cache_resource
def start_program():

    dataset = Preprocessing()
    dataset.readcsv("SmartMatch/Data/postings.csv")
    relevant_columns = ['title', 'location', 'skills_desc', 'description']
    dataset.combine_relevant_fields(relevant_columns)
    X = dataset.convert_postings_embeddings()


    return X, dataset


def processing(X, dataset, k):
        
        # instance of smartmatch class 
        running = SmartMatch(dataset)

        # Store cv data in user_cv
        user_cv = dataset.upload_file()

        # Processing can continue if the user uploads their CV. 
        if user_cv is not None:

            # Vectorize the user cv using sentence embeddings
            Y = dataset.convert_user_embeddings(user_cv)

            # Similarity scores between user query and job postings computed using cosine similarity
            Similarity_Scores = running.similarity_score(X, Y)
            
            # Top K job recommendations retrieved and outputted to user. 
            running.top_recommendations(k,Similarity_Scores)



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
    


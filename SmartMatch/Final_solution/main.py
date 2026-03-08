from preprocessing import Preprocessing 
import streamlit as st
from SmartMatchBaseline import Baseline 


# Caches preprocessing (data loading, feature extraction, TF-IDF fitting) to
# avoid recomputation, reducing execution time. 
@st.cache_data
def start_program():
    dataset = Preprocessing()
    st.title("SmartMatch")

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


# Execure the program
X, dataset = start_program()
processing(X, dataset, 3)



# Things to do later (written 8/12/2025) 
# - Measure execution time of the algorithm (done for preprocesing for performance testing on streamlit)
# - Add byte-encoder scoring for semantic reasoning (BERT, berta etc.)
# - Experiment with dataset size for optimal performance (quote in report)
# - Randomize dataset before sampling top 300 rows if >75,000 entries
# - For data cleaning, drop roles with NaN values instead of filling the spaces (.fillna)

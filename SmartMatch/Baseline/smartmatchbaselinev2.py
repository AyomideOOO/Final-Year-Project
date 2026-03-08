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

# time used for working out run-time
import time 

# streamlit used for the user interface
import streamlit as st 

# pdfplumber used for reading users cv into python for processing
import pdfplumber


from docx import Document


class SmartMatchBaseline:

    # Constructor: initializes the SmartMatchBaseline object
    # vectorizer: class used for vectorization 
    # df: the original dataset
    # pf: DataFrame containing combined relevant fields from df
    # X: TF-IDF vectors of the relevant fields
    # Y: TF-IDF vector of the user query

    def __init__(self):

        self.vectorizer = TfidfVectorizer()
        self.df = pd.DataFrame()
        self.X = None
        self.Y = None
        self.pf = None

    
    # 'readcsv' reads the provided csv file. 
    #  If the size of the file is greater than 75000, Dataframe 'df' will contain the rows.
    #  If not the whole csv will be read into 'df'

    def readcsv (self, my_csv ):
        self.df = pd.read_csv(my_csv)
        if len(self.df) > 75000:
            self.df = self.df.head(75000)


    def combine_relevant_fields (self, relevant_columns):

        # Create a new DataFrame 'pf' with one column 'Relevant_Fields'
        self.pf = pd.DataFrame(columns= ['Relevant_Fields'])

       
        # Data cleaning: fillna stores NaN values as empty strings.
        # this is to prevent data loss from concatinating rows. 

        # Store the first relevant column from 'df' into 'pf', converting text to lowercase
        self.pf['Relevant_Fields'] = self.df[relevant_columns[0]].fillna('').str.lower()

        
        # For loop stores the remaining relevant columns from 'df' into 'pf' 
        # concatenating each column value with a comma.
        for i in range(1,len(relevant_columns)):
            self.pf['Relevant_Fields'] += ", " + self.df[relevant_columns[i]].fillna('').str.lower() 

          

    def convert_to_vector(self):
        # vectorizes the column 'Relevant_Fields' in dataframe 'pf' using TF-IDF
        self.X = self.vectorizer.fit_transform(self.pf['Relevant_Fields'])
        return self.X
    

    def user_vector(self, user_input):
        # vectorize the user input
        self.Y = self.vectorizer.transform([user_input])
        return self.Y
    

    # NEED TO REWRITE COMMENT HERE ON A LATER DAY
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
            st.text(f" {i+1}) {self.df['title'][index]} \nSimilarity Score: {scores[index]} \n\n {self.df['description'][index]} \n\n",)

        
    

    # next session: Use hash-map to combine indexes with their respective cosine_similarity values
    # make a graph and plot the x axis -> the cosine similarities + role names, y axis -> 0 -to 1. 
    # this would show the relationship in the similarity and why certain roles showed in top k recommendations

    def graph_representation(self,k):
        similarity_score = self.similarity_score().flatten()
        best = np.sort(similarity_score)[-k:][::-1]
        print(best)

    

# Function prompts user to input their CV
# CV is read into python as a pdf, where the text is extracted and returned. 

def upload_file():

    text = ""
    uploaded_file = st.file_uploader("Upload your CV for personalised job recommendations", type = ['pdf', 'docx'] ) 
    if uploaded_file is not None:


        # checks if the 'type' of the file matches the standard for pdf files.
        if uploaded_file.type == 'application/pdf':
            with pdfplumber.open(uploaded_file) as pdf:

                # loop to extract the text from each page within the pdf
                # if the page is empty, appends "" to the string 'text'.
                for page in pdf.pages:
                    text += page.extract_text() or ""
                text = text.lower()

        # checks if the 'type' of the file matches the standard for word documents
        elif uploaded_file.type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
            document = Document(uploaded_file)
            for p in document.paragraphs:
                text += p.text
            text = text.lower()


        st.success(f'Successfully Uploaded file: {uploaded_file.name}')
        return text


# Caches preprocessing (data loading, feature extraction, TF-IDF fitting) to
# avoid recomputation, reducing execution time. 
@st.cache_data
def start_program():

    dataset = SmartMatchBaseline()
    st.title("SmartMatch")

    dataset.readcsv("SmartMatch/Data/postings.csv")
    relevant_columns = ['title', 'location', 'skills_desc', 'description']
    dataset.combine_relevant_fields(relevant_columns)

    X = dataset.convert_to_vector()
    return X, dataset



def processing(X, dataset, k):

    # Store cv data in user_cv
    user_cv = upload_file()

    # Processing can continue if the user uploads their CV. 
    if user_cv is not None:

        # Vectorize the user cv using TF-IDF 
        Y = dataset.user_vector(user_cv)

        # Similarity scores between user query and job postings computed using cosine similarity
        Similarity_Scores = dataset.similarity_score(X,Y)

        # Top K job recommendations retrieved and outputted to user. 
        dataset.top_recommendations(k,Similarity_Scores)

    # User enters the type of role they are looking for 
    # user_input = st.text_input("Enter the type of role you are looking for: ")

    #if user_input != "":
        
        # User query is vectorized using TF-IDF
        # Y = dataset.user_vector(user_input)
    
         
        # Similarity_Scores = dataset.similarity_score(X,Y)

        # Top K job recommendations retrieved and outputted to user. 
        # dataset.top_recommendations(k,Similarity_Scores)


####   Preprocessing   ####

# Start of preprocessing time 
start = time.time()
X, dataset = start_program()


# calculating preprocessing time 
end = time.time()
Preprocessing_duration = end - start 
print(f'Elapsed time of preprocessing: {Preprocessing_duration}')


####   Processing   ####
processing(X, dataset, 3)


####   End of Execution   ####

# Things to do later (written 8/12/2025) 
# - Measure execution time of the algorithm (done for preprocesing for performance testing on streamlit)
# - Add byte-encoder scoring for semantic reasoning (BERT, berta etc.)
# - Experiment with dataset size for optimal performance (quote in report)
# - Randomize dataset before sampling top 300 rows if >75,000 entries
# - For data cleaning, drop roles with NaN values instead of filling the spaces (.fillna)

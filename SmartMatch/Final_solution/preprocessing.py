
# Pandas used for storing csvs as dataframes: df, pf 
import pandas as pd

# Implemented TF-IDF for vectorizing the user input
# and the 'relevant fields' dataframe.
from sklearn.feature_extraction.text import TfidfVectorizer

# time used for working out run-time
import time 

# pdfplumber used for reading users cv into python for processing
import pdfplumber

from docx import Document

import streamlit as st

# Implemented to check whether the path, to the saved job postings embeddings exists.
import os

import numpy as np

# implemented for semanticly represented vectors using sentence embeddings. 
from sentence_transformers import SentenceTransformer 

# Used for efficient storage and loading of TF-IDF sparse matrices, to reduce recomputation time
from scipy import sparse

# Used to store the vectorizer, to reduce recomputation time
import joblib

class Preprocessing:

    def __init__(self):

        self.vectorizer = TfidfVectorizer()
        self.df = pd.DataFrame()
        self.X = None
        self.Y = None
        self.pf = None
        self.model = None
        self.uploaded_file = None

    
    def readcsv (self, my_csv ):
        self.df = pd.read_csv(my_csv)
        if len(self.df) > 75000:
            self.df = self.df.head(75000)

    
    def combine_relevant_fields (self, relevant_columns):

        # Create a new DataFrame 'pf' with one column 'Relevant_Fields'
        self.pf = pd.DataFrame(columns= ['Relevant_Fields'])

        # Assigns the job_id associated to each role. 
        self.pf["job_id"] = self.df["job_id"]

        # Data cleaning: fillna stores NaN values as empty strings.
        # this is to prevent data loss from concatinating rows. 

        # Store the first relevant column from 'df' into 'pf', converting text to lowercase
        self.pf['Relevant_Fields'] = self.df[relevant_columns[0]].fillna('').str.lower()
        
        # For loop stores the remaining relevant columns from 'df' into 'pf' 
        # concatenating each column value with a comma.
        for i in range(1,len(relevant_columns)):
            self.pf['Relevant_Fields'] += ", " + self.df[relevant_columns[i]].fillna('').str.lower() 

    

    # function that returns the sentence embedder model used
    def get_model(self):
        if self.model == None:
            self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        return self.model
    

    
    # TF-IDF Vectorisation (Job-postings)
    def convert_postings_TFIDF(self):

        # file path where the precomputed TF-IDF matrix is stored
        matrix_path = "SmartMatch/Data/job_tfidf.npz"

        # file path where the precomputed vectorizer is stored
        vectorizer_path = "SmartMatch/Data/tfidf_vectorizer.pkl"

        # if file exists, load the saved TF-IDF matrix instead of recomputing it
        if os.path.exists(matrix_path) and os.path.exists(vectorizer_path):

            # load stored sparse matrix (job postings already vectorised)
            self.X = sparse.load_npz(matrix_path)

            # load the vectorizer
            self.vectorizer = joblib.load(vectorizer_path)

        # If the file does not exist in that path, job postings is vectorised and stored in file 
        # to reduce execution time
        else:

            # fit TF-IDF on job postings and convert text into vectors
            self.X = self.vectorizer.fit_transform(self.pf['Relevant_Fields'])

            # save the resulting sparse matrix to avoid recomputating next time
            sparse.save_npz(matrix_path, self.X)

            # saves the resulting vectorizer to avoid recomputating next time
            joblib.dump(self.vectorizer, vectorizer_path)

        return self.X
    
    
    # TF-IDF Vectorization (User CV)
    def convert_user_TFIDF(self,user_input):
        # vectorize the user input
        self.Y = self.vectorizer.transform([user_input])
        return self.Y
    
    
    # Sentence Embeddings Vectorization (Job-postings)
    def convert_postings_embeddings(self):

        # stores location of the precomputed embedded job postings
        embeddings_path = "SmartMatch/Data/job_embeddings.npy"

        # loads the stored embedded job postings into X if the path to the file exists
        if os.path.exists(embeddings_path):
            self.X = np.load(embeddings_path)

        # If the file does not exist in that path, job postings is embedded and stored in file 
        # to reduce execution time
        else:

            # gets the current model being used 
            model = self.get_model()

            # model encodes the job postings converting it to a dense matrix
            self.X = model.encode(self.pf['Relevant_Fields'], show_progress_bar = True)

            # saves the embedded job postings into binary file in NumPy format 
            np.save(embeddings_path, self.X)
        return self.X
    

    # Sentence Embeddings Vectorization (User CV) 
    # NOTE: dont need check if model is none as this step happens after.
    def convert_user_embeddings(self,user_input):
        model = self.get_model()
        self.Y = model.encode([user_input])
        return self.Y
    

    
    def upload_file(self):
        text = ""
        self.uploaded_file = st.file_uploader("Upload your CV for personalised job recommendations", type = ['pdf', 'docx'] ) 
        if self.uploaded_file is not None:

        # checks if the 'type' of the file matches the standard for pdf files.
            if self.uploaded_file.type == 'application/pdf':
                with pdfplumber.open(self.uploaded_file) as pdf:

                # loop to extract the text from each page within the pdf
                # if the page is empty, appends "" to the string 'text'.
                    for page in pdf.pages:
                        text += page.extract_text() or ""
                    text = text.lower()

        # checks if the 'type' of the file matches the standard for word documents
            elif self.uploaded_file.type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
                document = Document(self.uploaded_file)
                for p in document.paragraphs:
                    text += p.text
                text = text.lower()

            st.success(f'Successfully Uploaded file: {self.uploaded_file.name}')
            return text
        
         # if there is no uploaded file
        return None 
    

#test = Preprocessing()
#test.readcsv("SmartMatch/Data/postings.csv")

#test.combine_relevant_fields(['title', 'location', 'skills_desc', 'description'])


# testing to check whether the job_id in the original dataset (df), aligns with
# the new dataframe pf. 
#check = (
#    test.df["job_id"].reset_index(drop=True) ==
#    test.pf["job_id"].reset_index(drop=True)
#).all()

#print(check)

#print(test.pf.head(5))










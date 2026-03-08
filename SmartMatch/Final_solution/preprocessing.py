
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



class Preprocessing:

    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.df = pd.DataFrame()
        self.X = None
        self.Y = None
        self.pf = None

    
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
    
    
    # TF-IDF Vectorization (Job-postings)
    def convert_postings_TFIDF(self):
        # vectorizes the column 'Relevant_Fields' in dataframe 'pf' using TF-IDF
        self.X = self.vectorizer.fit_transform(self.pf['Relevant_Fields'])
        return self.X
    
    
    # TF-IDF Vectorization (User CV)
    def convert_user_TFIDF(self,user_input):
        # vectorize the user input
        self.Y = self.vectorizer.transform([user_input])
        return self.Y
    
    
    # Sentence Embeddings Vectorization (Job-postings)
    def convert_postings_embeddings(self):
        pass

    # Sentence Embeddings Vectorization (User CV)
    def convert_user_embeddings(self):    
        pass
    
    
    def upload_file(self):
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


    







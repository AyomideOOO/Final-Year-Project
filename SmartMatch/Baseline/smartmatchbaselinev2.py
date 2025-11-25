# Baseline v2
# Author: Ayomide Osineye
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from heapq import nlargest

class SmartMatchBaseline:


    # constructor 
    def __init__(self):
        self.df = None
        self.X = None
        self.Y = None
        self.pf = None


    def readcsv (self, my_csv ):
        self.df = pd.read_csv(my_csv)
    

    def combine_relevant_fields (self, relevant_columns, my_csv):
        self.pf = pd.DataFrame(columns= ['Relevant_Fields'])

        # combine all the fields in relevant_columns into relevant_fields
        # for each role.

        self.pf['Relevant_Fields'] = self.df[relevant_columns[0]]

        for i in range(1,len(relevant_columns)):
            self.pf['Relevant_Fields'] += ", " + self.df[relevant_columns[i]].astype(str) 
        self.pf.to_csv(my_csv, index= False)


    def convert_to_vectors(self):

        # convert the the relevant fields to vector values using TF-IDF
        vectorizer = TfidfVectorizer()
        self.X = vectorizer.fit_transform(self.pf['Relevant_Fields'])

        # ask the user for what placement they are looking for 
        user_input = input("Enter The Type of Placement you are looking for: ")

        # vectorize the user input
        self.Y = vectorizer.transform([user_input])

        return self.X, self.Y
    

    def similarity_score(self):
       X, Y = self.convert_to_vectors()
       similarity_score = cosine_similarity(X,Y)
    
       return similarity_score
    

    def top_recommendations(self,k):
        similarity_score = self.similarity_score().flatten()
        k_best = np.argsort(similarity_score)[-k:][::-1]

        n = len(k_best)

        for i in range(n):
            print(f'{i+1}) {self.df['Job_title'][k_best[i]]}')

      

# Testing out class
new = SmartMatchBaseline()
new.readcsv("baseline_dataset.csv")


# get the relevant columns = 
relevant = ['Job_title', 'Location', 'Skills', 'Industry']
new.combine_relevant_fields(relevant, 'new_test.csv')



# top recommendations
new.top_recommendations(2)







    

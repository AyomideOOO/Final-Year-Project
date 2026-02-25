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


class SmartMatchBaseline:

    # Constructor: initializes the SmartMatchBaseline object
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
        if len(pd.read_csv(my_csv)) > 75000:
            self.df = pd.read_csv(my_csv).head(75000)
        else:
            self.df = pd.read_csv(my_csv)


    def combine_relevant_fields (self, relevant_columns, my_csv):

        # Create a new DataFrame 'pf' with one column 'Relevant_Fields'
        self.pf = pd.DataFrame(columns= ['Relevant_Fields'])

       
        # Data cleaning: fillna stores NaN values as empty strings.
        # this is to prevent data loss from concatinating rows. 

        # Store the first relevant column from 'df' into 'pf', converting text to lowercase
        self.pf['Relevant_Fields'] = self.df[relevant_columns[0]].fillna("").str.lower()

        
        # For loop stores the remaining relevant columns from 'df' into 'pf' 
        # concatenating each column with a comma.

        for i in range(1,len(relevant_columns)):
            self.pf['Relevant_Fields'] += ", " + self.df[relevant_columns[i]].fillna("").str.lower() 


        # Stores new Dataframe 'pf' as a CSV file.
        self.pf.to_csv(my_csv, index= False)


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
            print(f" {i+1}) {self.df['title'][index]} \n\n {self.df['description'][index]} \n\n")
    
     
    # next session: Use hash-map to combine indexes with their respective cosine_similarity values
    # make a graph and plot the x axis -> the cosine similarities + role names, y axis -> 0 -to 1. 
    # this would show the relationship in the similarity and why certain roles showed in top k recommendations

    def graph_representation(self,k):

        similarity_score = self.similarity_score().flatten()
        best = np.sort(similarity_score)[-k:][::-1]

        print(best)
      


# Preprocessing      
real_dataset = SmartMatchBaseline()
running = True 
start = 0

while len(real_dataset.df) == 0:
    start = time.time()
    
    real_dataset.readcsv('Data/postings.csv')

    # get all the relevant fields together into new dataframe
    relevant_columns = ['title', 'location', 'skills_desc', 'description']
    real_dataset.combine_relevant_fields(relevant_columns, 'Data/relevant_fields.csv')
    
    

# Vectorizes the dataframe (pf)
X = real_dataset.convert_to_vector()

# calculating preprocessing time 
end = time.time()
Preprocessing_duration = end - start 
print(f'Elapsed time of preprocessing: {Preprocessing_duration}')



# Processing
# Cosine Similarity + Top K recommendations
while running: 

   # ask the user for what role they are looking for 
    user_input = input("Enter the type of role you are looking for: ")
    Y = real_dataset.user_vector(user_input)
    Similarity_Scores = real_dataset.similarity_score(X,Y)
    real_dataset.top_recommendations(3,Similarity_Scores)
    user_input = input("Enter (Y) to find another role, Enter (N) to stop: ").lower()
    if user_input == 'Y'.lower():
       running = True
    elif user_input == 'N'.lower():
        print("End. ")
        running = False

        
        








# Things to do later (written 8/12/2025) 
# - Measure execution time of the algorithm
# - Add byte-encoder scoring for semantic reasoning (BERT, berta etc.)
# - Experiment with dataset size for optimal performance (quote in report)
# - Randomize dataset before sampling top 300 rows if >75,000 entries
# - For data cleaning, drop roles with NaN values instead of filling the spaces (.fillna)

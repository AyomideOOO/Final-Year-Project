# Baseline 

# imports
import pandas as pd 
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Load dataset 
df = pd.read_csv("baseline_dataset.csv", encoding="latin1")

# create seperate lists for columns 

# vectorization via TF-IDF
vectorizer = TfidfVectorizer()

# im vectorising only based on Job_title, Location, Skills, Industry
# create a new csv for combined relevant fields for vectorisation

# pf = pd.DataFrame(columns=['Relevant_Fields'])
# pf['Relevant_Fields'] = df['Job_title'] +", "+df['Location'] +", "+ df['Skills'] +", "+ df['Industry']
# pf.to_csv('relevant_fields.csv', index=False)


# make a new dataframe for relevant_fields csv file
pf = pd.read_csv("relevant_fields.csv")


# Apply TF-IDF Vectorization

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(pf['Relevant_Fields'])


# Ask the user to enter the role they are looking for 
user_input = input("Enter the type of placement you are looking for :): ")


# Apply TF-IDF Vectorization
Y = vectorizer.transform([user_input])



# Apply Cosine Similarity
similarity_scores = cosine_similarity(X,Y)
print(similarity_scores)


# Get the most relevant placement based on cosine similariy

best_index = similarity_scores.argmax()
print(best_index)


# returns the best placement based on user preference
print(f'Based on your choice, the best placement is:{df['Job_title'][best_index]}')







# class for my evaluation

from preprocessing import Preprocessing
import importlib
import numpy as np
import pandas as pd
import pdfplumber 
from sklearn.metrics.pairwise import cosine_similarity
from docx import Document


# returns roles containing key words in order to make my dictionary
# Law
#print(testing.df[testing.df['title'].str.contains('Law|legal|attorney|compliance', case=False, na=False)]['title'].value_counts().head(20))
#print("#######################################", '\n')

# Finance
#print(testing.df[testing.df['title'].str.contains('Financial|Financial Advisor|Audit|Accountant|Bank|Investment|Finance|Controller', case=False, na=False)]['title'].value_counts().head(20))
#print("#######################################", '\n')

# Technology
#print(testing.df[testing.df['title'].str.contains('Technology|Computer Science', case=False, na=False)]['title'].value_counts().head(20))
#print("#######################################", '\n')

# Operations
#print(testing.df[testing.df['title'].str.contains('Operations|Business Management|Project Management|Business Operations|Logistics|Operations Coordinator|coordinator', case=False, na=False)]['title'].value_counts().head(20), '\n')
#print("#######################################", '\n')

#Engineering
#print(testing.df[testing.df['title'].str.contains('electrical engineer|mechanical engineer|civil engineer|hardware engineer|engineering manager', case=False, na=False)]['title'].value_counts().head(20))



# Creating dictionary for mapping CV sectors, with the associated names within the dataset
# to manually create the classification links between recommended roles and the sector they belong to.


sectors_dict = {
    'Law': ['attorney', 'legal', 'paralegal', 'law'],

    'Finance': ['financial advisor', 'audit', 'accountant', 'bank', 'investment', 'controller', 'finance'],

    'Tech':['software engineer', 'data analyst', 'senior software engineer',
            'data engineer', 'quality engineer', 'software developer',
            'frontend developer', 'back end developer', 'web developer',
            'information technology', 'technical lead',
            'chief technology officer', 'it manager',
            'systems administrator', 'it support', 'developer', 'full stack engineer'],

    'Operations':['operations manager', 'operations assistant manager', 'director of operations',
               'project coordinator', 'operations supervisor', 'logistics coordinator',
               'operations coordinator', 'program coordinator', 'logistics specialist', 
               'operations associate'],

    'Engineering':['electrical engineer', 'mechanical engineer', 'civil engineer',
                'hardware engineer', 'engineering manager', 'manufacturing engineer',
                'structural engineer', 'chemical engineer', 'industrial engineer']
}

# creating a lookup table for cvs and the their class (sector)
cv_and_sectors = {
    'AyomideCurrentCV_cv.pdf': 'Tech',
    'AyomideOLD_cv.pdf': 'Tech',
    'Grace_cv.pdf': 'Tech',
    'Ade_cv.pdf': 'Engineering',
    'Brandon_cv.docx': 'Finance',
    'Joel_cv.pdf': 'Finance',
    'Keisha_cv.pdf': 'Law',
    'Michael_cv.docx': 'Engineering',
    'Nathaniel_cv.pdf': 'Operations',
    'Zainab_cv.pdf': 'Finance'
    }


# testing, evaluating my 10 cvs for each model, getting the accuracy per cv sector
testing = Preprocessing()
testing.readcsv("SmartMatch/Data/postings.csv")


##   TF-IDF model accuracy   ##
testing.combine_relevant_fields(['title', 'location', 'skills_desc', 'description'])



# Will store the string user_cv 
text = " "
cv = "Brandon_cv.docx"

if  cv.endswith(".pdf"):
    with pdfplumber.open(f'SmartMatch/Sample_cvs/{cv}') as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
            text = text.lower()

elif cv.endswith(".docx"):
    document = Document(f'SmartMatch/Sample_cvs/{cv}')
    for p in document.paragraphs:
        text += p.text
        text = text.lower()


# Apply TF-IDF to postings and User CV
X = testing.convert_postings_TFIDF()
Y = testing.convert_user_TFIDF(text)

# Calculating the similarity scores between the postings and the user cv 
similarity_score = cosine_similarity(X,Y)


# Flatten the similarity_score array 2-D -> 1-D to allow for easier
# sorting. 
scores = similarity_score.flatten()
k = 3
count = 0

# returns 'k' indexs of the roles with the highest similarity scores within the dataset. 
# argsort sorts the scores in ascending order, maining the index which it was stored in. 
k_best = np.argsort(scores)[-k:][::-1]
print(k_best)


# printing out the names of the roles with the best similarity scores
#for i in range(len(k_best)):
#    print("\n" + testing.df['title'][k_best[i]] + "\n")

# store recommended titles in a list
recommended_titles = []
n = len(k_best)

for i in range(n):
    temp = testing.df['title'][k_best[i]]
    recommended_titles.append(temp)
print(recommended_titles)




get_sector_values = cv_and_sectors["Brandon_cv.docx"]

# check each title against the sector keywords
for title in recommended_titles:

    # loops through each value within the specified sector (key) in sectors_dict
    for value in sectors_dict[get_sector_values]:

        # checks if the value retrieved from the dictionary, is contained in the job postings title.
        if value in title:
            count += 1

            # breaks to consider cases where the 
            # recommended role appears mutliple times in the dictionary
            break


print("Percentage of correctly identified sectors that relate to the user cv: ", ((count/k) * 100))



# Changing Strategy: Applying semantic supervised learning. rather than lexical matching  which is inefficient as you values within the dictionary have 
# to be exactly the same as the name in the job postings, Essentially being another word based match making which is not appropriate for my project



        












    























































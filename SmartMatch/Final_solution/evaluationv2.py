# Evaluation class where firstly i will be doing semantic classification to find the correcttly classified roles returned
# that also represent the same sector as the user cv

from preprocessing import Preprocessing
import importlib
import numpy as np
import pandas as pd
import pdfplumber 
from sklearn.metrics.pairwise import cosine_similarity
from docx import Document
# implemented for semanticly represented vectors using sentence embeddings. 
from sentence_transformers import SentenceTransformer 






















model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# semantic dictionary which will be used to embedd the values within the structure
# Will embedd each value within the key-value pair and apply cosine similarity between that
# each job posting recommended, assigning the sector to the job based on the highest similarity score.

sectors = {
    'Law': "Legal jobs including attorneys, paralegals, compliance and corporate law roles.",

    'Finance': "Finance jobs including accounting, auditing, banking, investment analysis, financial planning.",

    'Tech':"Technology jobs including software engineering, data science, machine learning, IT support, web development, AI engineering.",

    'Operations': "Operations jobs including project management, logistics, coordination and business operations.",
             
    'Engineering': "Engineering jobs including mechanical, civil, electrical, manufacturing and industrial engineering."
}

# embed each value-pair individually





# creating a lookup table for cvs and the their class (sector)
cv_and_sectors = {
    'AyomideCurrent_cv.pdf': 'Tech',
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




testing = Preprocessing()
testing.readcsv("SmartMatch/Data/postings.csv")

# vectorize job postings (TF-IDF)
testing.combine_relevant_fields(['title', 'location', 'skills_desc', 'description'])
X = testing.convert_postings_TFIDF()



k = 5
count = 0
match_scores = None

for key in cv_and_sectors.keys():
    text = ""
    
    if key.endswith(".pdf"):
          with pdfplumber.open(f"SmartMatch/Sample_cvs/{key}") as pdf:
               # loop to extract the text from each page within the pdf
               # if the page is empty, appends "" to the string 'text'.
            for page in pdf.pages:
                text += page.extract_text() or ""
                text = text.lower()
    
    elif key.endswith(".docx"):
        document = Document(f"SmartMatch/Sample_cvs/{key}")
        for p in document.paragraphs:
            text += p.text
            text = text.lower()

    # vectorize user cv (TF-IDF)
    Y = testing.convert_user_TFIDF(text)

    # Generate similarity scores between postings and cv
    similarity_score = cosine_similarity(X,Y)

    # Flatten the similarity_score array 2-D -> 1-D to allow for easier sorting. 
    scores = similarity_score.flatten() 

    # returns 'k' indexs of the roles with the highest similarity scores within the dataset. 
    # argsort sorts the scores in ascending order, maining the index which it was stored in. 
    k_best = np.argsort(scores)[-k:][::-1]

    # gets top k roles
    # store recommended titles in a list
    recommended_titles = []
    n = len(k_best)
    for i in range(n):
        temp = testing.df['title'][k_best[i]]
        recommended_titles.append(temp)
  
    # sector classification begins # 
    # this is done for each role recommended per cv against each embedded sector


    # will be used to get the sector the role has been mapped to based
    # on the highest similarity score between the role in the iteration all the 
    # different sector information. 
    sector_names = list(sectors.keys())

    # stores information related to each sector and stores it in a list
    sector_values = list(sectors.values())


    # the list is encoded which will be used to work out the similarity score for
    # each role against all sectors
    encoded_sector_desc = model.encode(sector_values)


    for title in recommended_titles:
     # NOTE: stores the encoded job titles in a new variable
     # used [] around the input so it is 2-D as originally just a string.
     encoded_titles = model.encode([title])

         
    # get the similarity scores between the job and the sector desc
     score = cosine_similarity(encoded_titles, encoded_sector_desc)

     # NOTE: item() coverts numpy array to scalar value
     match_scores = score.flatten()

    # gets the index of the highest score, which is the derived sector for that role.
     label = np.argmax(match_scores)
     predicted_sector = sector_names[label]
     print(f"For CV {key}, ROLE {title}, sector classified: {predicted_sector}")
    print("\n")
    


   





    
# print(match_scores.shape)



     








    








            









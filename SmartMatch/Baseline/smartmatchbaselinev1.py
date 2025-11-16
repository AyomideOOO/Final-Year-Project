# Baseline 

# imports
import pandas as pd 
from sklearn.feature_extraction.text import TfidfVectorizer


# Load dataset 
df = pd.read_csv("baseline_dataset.csv", encoding="latin1")

# create seperate lists for columns 



# vectorization via TF-IDF
vectorizer = TfidfVectorizer()

print(df.loc[0])




# Baseline v2
# Author: Ayomide Osineye
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class SmartMatchBaseline:


    # constructor 
    def __init__(self):
        pass 


    def readcsv (self, my_csv ):
        df = pd.read_csv(my_csv)
        return df
    

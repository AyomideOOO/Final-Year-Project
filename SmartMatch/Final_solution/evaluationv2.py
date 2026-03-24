# Evaluation class where firstly i will be doing semantic classification to find the correcttly classified roles returned
# that also represent the same sector as the user cv

from preprocessing import Preprocessing
import pandas as pd
import pdfplumber 
from sklearn.metrics.pairwise import cosine_similarity
from docx import Document
# implemented for semanticly represented vectors using sentence embeddings. 
from sentence_transformers import SentenceTransformer 

import numpy as np


# imported for visualizing my evaluations
import matplotlib

# TkAgg is a Matplotlib backend which renders my plots using 
# a GUI window i can interact with
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt

class Evaluation:

    # constructor 
    def __init__(self):
        self.preprocessor = Preprocessing()
        self.preprocessor.readcsv("SmartMatch/Data/postings.csv")
        self.preprocessor.combine_relevant_fields(['title', 'location', 'skills_desc', 'description'])


    def get_recommended_roles(self,k_best):
        recommended_titles = []
        n = len(k_best)
        for i in range(n):
            temp = self.preprocessor.df['title'][k_best[i]]
            recommended_titles.append(temp)
        return recommended_titles
    

    def cv_sector_dict(self):
        cv_and_sectors = {
            'Mide_cv.pdf': 'Tech',
            'MideOLD_cv.pdf': 'Tech',
            'Grace_cv.pdf': 'Tech',
            'Ade_cv.pdf': 'Engineering',
            'Brandon_cv.docx': 'Finance',
            'Joel_cv.pdf': 'Finance',
            'Keisha_cv.pdf': 'Law',
            'Michael_cv.docx': 'Engineering',
            'Nathaniel_cv.pdf': 'Operations',
            'Zainab_cv.pdf': 'Finance'
            }
        return cv_and_sectors
    

    def sectors_desc(self):
        sectors = {
            'Law': "Legal jobs including attorneys, paralegals, compliance and corporate law roles.",
            'Finance': "Finance jobs including accounting, auditing, banking, investment analysis, financial planning.",
            'Tech':"Technology jobs including software engineering, data science, machine learning, IT support, web development, AI engineering.",
            'Operations': "Operations jobs including project management, logistics, coordination and business operations.",
            'Engineering': "Engineering jobs including mechanical, civil, electrical, manufacturing and industrial engineering."
            }
        return list(sectors.keys()), list(sectors.values())
    
    
    # Function for running tf-idf 
    def run_tfidf(self):

        # stores the sector names and values in seperate lists.
        # sector_names will be used to predict/classify the sector to the role
        # sector_values will be used for semantic embeddings and compared with each role recommended to the user. 
        sector_names, sector_values = self.sectors_desc()
        
        # The model implemeted to embed the sector desc for the classification. 
        # Retrieved the model from preprocessing for consistency.
        model = self.preprocessor.get_model()


        # the list is encoded which will be used to work out the similarity score for
        # each role against all sectors
        encoded_sector_desc = model.encode(sector_values)

        # stores precision of K roles recommended per CV where 
        # each index represents the results for a different CV.
        all_precisions = []

        # the number of roles returned to each user.
        k = 5
        
        # vectorize job postings
        X = self.preprocessor.convert_postings_TFIDF()

        # stores cv dictionary in cv_dict for processing.
        cv_dict = self.cv_sector_dict()

        for key in cv_dict.keys():
            count = 0
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
            Y = self.preprocessor.convert_user_TFIDF(text)

            # Compute similarity score betwwen job postings and CV
            similarity_score = cosine_similarity(X,Y)

            # Flatten the similarity_score array 2-D -> 1-D to allow for easier sorting. 
            scores = similarity_score.flatten() 

            # returns 'k' indexs of the roles with the highest similarity scores within the dataset. 
            # argsort sorts the scores in ascending order, maining the index which it was stored in. 
            k_best = np.argsort(scores)[-k:][::-1]

            # gets top k roles
            # store recommended titles in a list
            recommended_titles = self.get_recommended_roles(k_best)

            # sector classification begins # 
            # this is done for each role recommended per cv against each embedded sector
            for title in recommended_titles:

                # NOTE: stores the encoded job titles in a new variable
                # used [] around the input so it output is 2-D. as originally just a string.
                encoded_titles = model.encode([title])

                # get the similarity scores between the job and the sector desc
                score = cosine_similarity(encoded_titles, encoded_sector_desc)

                # flatten converts 2-D array -> 1-D array
                match_scores = score.flatten()

                # gets the index of the highest score, which is the derived sector for that role.
                label = np.argmax(match_scores)
                predicted_sector = sector_names[label]

                # Check if the predicted sector for that role is the same as the sector
                # belonging to the cv 
                if predicted_sector == cv_dict[key]:
                    count += 1
                print(f"For CV {key}, ROLE {title}, sector classified: {predicted_sector}")

            # Get the precision of ranked roles being within the same sector as the user cv
            precision = (count / k)
            all_precisions.append(precision)

            # return the precisison
            print(f"The precision of correctly indenfied roles within the same sector as the User CV: {precision}")
            print("\n")

        # return the average precision across all the cvs. 
        average_precision = sum(all_precisions) / len(all_precisions)
        print(f"The Overall precision:{average_precision * 100}%")

        return all_precisions, (average_precision * 100)
    

    # vectorize job postings
    def run_embeddings(self):

        # stores the sector names and values in seperate lists.
        # sector_names will be used to predict/classify the sector to the role
        # sector_values will be used for semantic embeddings and compared with each role recommended to the user. 
        sector_names, sector_values = self.sectors_desc()

        # The model implemeted to embed the sector desc for the classification, job postings and cvs.
        # retrieved the model from pre-processing.
        model = self.preprocessor.get_model()

        # the list is encoded which will be used to work out the similarity score for
        # each role against all sectors
        encoded_sector_desc = model.encode(sector_values)

        # stores precision of K roles recommended per CV where 
        # each index represents the results for a different CV.
        all_precisions = []

        # the number of roles returned to each user.
        k = 5

        # embed job postings
        X = self.preprocessor.convert_postings_embeddings()

        # stores cv dictionary in cv_dict for processing.
        cv_dict = self.cv_sector_dict()

        for key in cv_dict.keys():
            count = 0
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
            
            # vectorize user cv (embedding)
            Y = self.preprocessor.convert_user_embeddings(text)

            # Compute similarity score betwwen job postings and CV
            similarity_score = cosine_similarity(X,Y)

            # Flatten the similarity_score array 2-D -> 1-D to allow for easier sorting. 
            scores = similarity_score.flatten() 

            # returns 'k' indexs of the roles with the highest similarity scores within the dataset. 
            # argsort sorts the scores in ascending order, maining the index which it was stored in. 
            k_best = np.argsort(scores)[-k:][::-1]

            # gets top k roles
            # store recommended titles in a list
            recommended_titles = self.get_recommended_roles(k_best)

            # sector classification begins # 
            # this is done for each role recommended per cv against each embedded sector
            for title in recommended_titles:

                # NOTE: stores the encoded job titles in a new variable
                # used [] around the input so it output is 2-D. as originally just a string.
                encoded_titles = model.encode([title])

                # get the similarity scores between the job and the sector desc
                score = cosine_similarity(encoded_titles, encoded_sector_desc)

                # flatten converts 2-D array -> 1-D array
                match_scores = score.flatten()

                # gets the index of the highest score, which is the derived sector for that role.
                label = np.argmax(match_scores)
                predicted_sector = sector_names[label]

                # Check if the predicted sector for that role is the same as the sector
                # belonging to the cv 
                if predicted_sector == cv_dict[key]:
                    count += 1
                print(f"For CV {key}, ROLE {title}, sector classified: {predicted_sector}")
            
            # Get the precision of ranked roles being within the same sector as the user cv
            precision = (count / k)
            all_precisions.append(precision)

            # return the precisison
            print(f"The precision of correctly indenfied roles within the same sector as the User CV: {precision}")
            print("\n")

        # return the average precision across all the cvs. 
        average_precision = sum(all_precisions) / len(all_precisions)
        print(f"The Overall precision:{average_precision * 100}%")

        return all_precisions, (average_precision * 100)


    # function that generates graph representations of my results
    def precision_graph_models(self, tfidf, embeddings):

        # retrieves cvs and their corresponding sectors.
        cv_and_sectors = self.cv_sector_dict()
       
        # removes "_cv.pdf and or "_cv.docx" for each cv in a dictionary ror better X labels
        plot_cv = [i.replace('_cv.pdf', '').replace('_cv.docx', '') for i in cv_and_sectors.keys()]


        # anomylising the user cvs for ethics reasons
        new_plots = [plot_cv[i].replace(plot_cv[i],f"CV {i+1}") for i in range(len(plot_cv))]

        # list of the precisions per CVs' (TF-IDF)
        results_tfidf = tfidf

        # list of the precisions per CVs' (Sentence Embeddings)
        results_embeddings = embeddings

        x = np.arange(len(new_plots))
        w = 0.35

        plt.bar(x - w/2, results_tfidf, w, label = 'TF-IDF')
        plt.bar(x + w/2, results_embeddings, w, label = 'all-MiniLM-L6-v2')
        plt.xticks(x,new_plots)
        plt.xlabel('CVs')
        plt.ylabel('Precision (0-1)')
        plt.title('Precision of Sector Classification for Top-K recommendations')
        plt.legend()

        # prevents labels being clipped when saving the image
        plt.savefig("SmartMatch/Graphs/precision_per_cv.png", dpi = 'figure')
        plt.show()
    
    # function that generates mean precision results 
    def mean_precision_graph(self, mean_tfidf, mean_embeddings):

        # retrieves cvs and their corresponding sectors.
        cv_and_sectors = self.cv_sector_dict()

        fig, ax = plt.subplots()

        models = ['TF-IDF' , 'all-MiniLM-L6-v2']
        results = [mean_tfidf, mean_embeddings]
        bar_labels = ['TF-IDF ', 'all-MiniLM-L6-v2']
        bar_colors = ['tab:red', 'tab:blue']

        ax.bar(models, results, label=bar_labels, color=bar_colors)


        ax.set_xlabel('Models')
        ax.set_ylabel('Mean Precision (%)')
        ax.set_title('Mean Precision per CV For Each Model')
        
        ax.legend()


        # prevents labels being clipped when saving the image
        plt.savefig("SmartMatch/Graphs/average_precision_per_model.png", dpi = 'figure')
        plt.show()

        




# testing
eval = Evaluation()
embeddings_model_per_cv, embeddings_mean = eval.run_embeddings()
tfidf_per_cv, tfidf_mean = eval.run_tfidf()
# eval.precision_graph_models(tfidf_per_cv, embeddings_model_per_cv)
eval.mean_precision_graph(tfidf_mean, embeddings_mean)







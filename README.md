# SmartMatch - Job Retrieval System

## Overview 

SmartMatch is a job retrieval system developed as a final year project. 
The project investigates and compares two approaches for retrieving job opportunities from a dataset:

1. A lexical retrieval approach using TF-IDF
2. A semantic retrieval approach using sentence embeddings

The aim was to explore how traditional keyword-based retrieval compares
with semantic similarity methods, and how different text representations influence the roles returned by the system.

## Features

- Dual retrieval pipeline:
  - TF-IDF based lexical retrieval
  - Sentence Transformer semantic retrieval

- Text preprocessing pipeline for job postings dataset and user CV input

- Ranking system that orders jobs based on similarity scores 

- Interactive Streamlit web interface to display both retrieval

- Cached/precomputed embeddings + vectorizer and job postings sparse matrix for improved runtime performance.


## Project Structure 
Final-Year-Project/

SmartMatch/
├── Data/
│   ├── job_embeddings.npy
│   ├── job_tfidf.npz
│   ├── postings.csv
│   ├── results.csv
│   ├── Resume.csv
│   └── tfidf_vectorizer.pkl
│
├── Final_solution/
│   ├── Main.py
│   ├── preprocessing.py
│   ├── evaluation.py
│   └── evaluationv2.py
│
├── pages/
│
├── Graphs/
│
├── Sample_cvs/

---

## Technologies Used 

- Python
- Streamlit
- Scikit-learn
- Sentence Transformers
- Pandas
- NumPy

---

## Retrieval Approaches

### TF-IDF (Lexical Approach)

It matches based on the keyword overlap between the user input and the job postings dataset.

Advantages:
- Fast processing
- Strong keyword matching

Limitations:
- Cannot understand meaning / semantics beyond exact words

---

### Semantic Retrieval 

The semantic pipeline uses sentence embeddings to encode the entire text input into a 
dense vector.

The system ranks jobs based on semantic similarity meaning even when different words are used within
their descriptions.

Advantages:
- Captures contextual meaning
- Better handles variations in wording

Limitations:
- More computationally expensive
- Longer runtime compared to TF-IDF

---

## Project Context

This project was completed as part of a final year dissertation investigating lexical and semantic approaches for job retrieval systems.

The evaluation focuses on comparing model behaviour and understanding how different text representations influence retrieval results. 
The comparison was performed in an unsupervised retrieval environment, focusing on differences between lexical and semantic approaches rather
than measuring recommendation accuracy. 

## Future improvements

Possible extensions include:

- User-based evaluation / user-interaction data
- Retrieval Augmented Generation pipeline implementation
- A dataset with ground truth labels for supervised evaluation between the models. 

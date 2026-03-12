import streamlit as st




# Landing Page Inital Content 

st.title("SmartMatch")
st.write("Welcome to SmartMatch — an AI-driven job recommendation system.")
st.write("Select a page from the sidebar to get started.")


# Navigation Panel 
pg = st.navigation([st.Page("pages/1_SmartMatchBaseline.py", title = "Baseline"), st.Page("pages/2_SmartMatch.py", title = "SmartMatch")])
pg.run()





# Things to do later (written 8/12/2025) 
# - Measure execution time of the algorithm (done for preprocesing for performance testing on streamlit)
# - Add byte-encoder scoring for semantic reasoning (BERT, berta etc.)
# - Experiment with dataset size for optimal performance (quote in report)
# - Randomize dataset before sampling top 300 rows if >75,000 entries
# - For data cleaning, drop roles with NaN values instead of filling the spaces (.fillna)

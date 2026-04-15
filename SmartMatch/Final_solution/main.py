import streamlit as st
# Landing Page Inital Content 

st.title("SmartMatch")
st.write("Welcome to SmartMatch — an AI-driven job recommendation system.")
st.write("Select a page from the sidebar to get started.")


# Navigation Panel 
pg = st.navigation([st.Page("pages/1_SmartMatchBaseline.py", title = "Baseline"), st.Page("pages/2_SmartMatch.py", title = "SmartMatch")])
pg.run()


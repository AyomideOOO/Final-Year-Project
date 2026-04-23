import streamlit as st

# Renames the tab page name
st.set_page_config(page_title="Overview")
st.title("SmartMatch Overview")

# text information for page
st.write(
    "SmartMatch is a comparative job recommendation system that evaluates two retrieval approaches: "
    "a TF-IDF based lexical model and a sentence embedding based semantic model."
)

st.write(
    "Each approach is implemented as a separate retrieval pipeline, allowing comparison of how "
    "different text representation methods influence job ranking results."
)

st.write("Use the sidebar to navigate between the two retrieval systems.")

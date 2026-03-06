# Understanding the implementation of steamlit

import streamlit as st

st.title("Hello World")



st.header("This is a header")
st.subheader("This is a subheader")
st.text("Hello my name is Ayomide")
st.markdown("### This is a markdown")


st.success("This was sucessful")

# exceptions
exp = ZeroDivisionError("Trying to divide by zero")
st.exception(exp)


# writing python into streamlit

st.write("Text with write")

for i in range(5):
    st.write(i)

# buttons

if st.button("Click ME"):
    st.text("Why you click me :(")


# user text input
name = st.text_input("Enter your name", "Type here...")

if st.button("Submit"):
    result = name.capitalize()
    st.success(result)


st.session_state['New data'] = 0


st.session_state["New data"] += 1

st.write(st.session_state)



import streamlit as st
from chatbot import run_complete_rag_pipeline

st.set_page_config(page_title="Course Catalog Assistant", page_icon="📚")

st.title("CS CheatSheet BOT")

query = st.text_input("Ask a question:")

if st.button("Send") and query:
    answer = run_complete_rag_pipeline(query)
    st.write("**Answer:**")
    st.write(answer)

import streamlit as st
import os
from PyPDF2 import PdfReader
import docx
from transformers import pipeline
import nltk
import requests
from bs4 import BeautifulSoup
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt')

# Set the page config at the top of the script
st.set_page_config(page_title="Chat With any Files", page_icon="💬", layout="wide")

# Load custom CSS
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def main():
    load_css("styles.css")
    
    st.title("💬 Chatbot with Document and Web URL Support")

    if "processComplete" not in st.session_state:
        st.session_state.processComplete = None

    with st.sidebar:
        st.header("Upload Files or Enter URL")
        uploaded_files = st.file_uploader("Upload your file", type=['pdf', 'docx'], accept_multiple_files=True)
        web_url = st.text_input("Or enter a web URL")
        process = st.button("Process")

    if process:
        if uploaded_files or web_url:
            with st.spinner("Processing..."):
                files_text = get_files_text(uploaded_files)
                web_text = get_web_text(web_url) if web_url else ""
                combined_text = files_text + web_text
                text_chunks = get_text_chunks(combined_text)
                st.session_state.text_chunks = text_chunks
                st.session_state.processComplete = True
                st.success("Files processed successfully!")
        else:
            st.warning("Please upload at least one file or enter a URL.")

    if st.session_state.processComplete:
        st.header("Ask a Question")
        user_question = st.text_input("Enter your question about the documents or URL:")
        if user_question:
            handle_user_input(user_question)

def get_files_text(uploaded_files):
    text = ""
    for uploaded_file in uploaded_files:
        split_tup = os.path.splitext(uploaded_file.name)
        file_extension = split_tup[1]
        if file_extension == ".pdf":
            text += get_pdf_text(uploaded_file)
        elif file_extension == ".docx":
            text += get_docx_text(uploaded_file)
    return text

def get_pdf_text(pdf):
    pdf_reader = PdfReader(pdf)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def get_docx_text(file):
    doc = docx.Document(file)
    allText = []
    for docpara in doc.paragraphs:
        allText.append(docpara.text)
    text = ' '.join(allText)
    return text

def get_web_text(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        paragraphs = soup.find_all('p')
        text = ' '.join([para.get_text() for para in paragraphs])
        return text
    except Exception as e:
        st.error(f"Error fetching the URL: {e}")
        return ""

def get_text_chunks(text, max_chunk_size=512):
    sentences = nltk.tokenize.sent_tokenize(text)
    chunks = []
    chunk = ""
    for sentence in sentences:
        if len(chunk) + len(sentence) <= max_chunk_size:
            chunk += sentence + " "
        else:
            chunks.append(chunk.strip())
            chunk = sentence + " "
    if chunk:
        chunks.append(chunk.strip())
    return chunks

def find_relevant_chunks(text_chunks, query, top_n=5):
    # Vectorize the text chunks and query using TF-IDF
    vectorizer = TfidfVectorizer().fit_transform([query] + text_chunks)
    vectors = vectorizer.toarray()

    query_vector = vectors[0].reshape(1, -1)
    chunk_vectors = vectors[1:]

    # Calculate cosine similarity
    similarities = cosine_similarity(query_vector, chunk_vectors).flatten()
    
    # Get top_n most similar chunks
    relevant_indices = similarities.argsort()[-top_n:][::-1]
    relevant_chunks = [text_chunks[i] for i in relevant_indices]
    return relevant_chunks

def clean_text(text):
    # Remove non-printable characters and other unwanted text artifacts
    printable = set("0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ ,.?!'-\n")
    return ''.join(filter(lambda x: x in printable, text))

def handle_user_input(user_question):
    text_chunks = st.session_state.text_chunks
    relevant_chunks = find_relevant_chunks(text_chunks, user_question)
    relevant_text = " ".join(relevant_chunks)  # Combine the most relevant chunks

    # Clean the relevant text
    cleaned_relevant_text = clean_text(relevant_text)

    # Initialize summarization model
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

    try:
        summary = summarizer(cleaned_relevant_text, max_length=150, min_length=40, do_sample=False)[0]['summary_text']
        st.subheader("Response:")
        st.markdown(f"{summary}")
    except Exception as e:
        st.write("Error during summarization:", str(e))

if __name__ == '__main__':
    main()

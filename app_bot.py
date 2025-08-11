import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI


#from dotenv import load_dotenv

from google.generativeai import GenerativeModel
import google.generativeai as genai

import os
os.environ["GOOGLE_API_KEY"] = "AIzaSyD9Plos33NA_JiasNE34FtswlXK2YJg30c"
def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return  text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():

    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    #model = ChatGoogleGenerativeAI(model="models/gemini-pro",api_version="v1",temperature=0.3)

    

    model = ChatGoogleGenerativeAI(
    model="models/gemini-1.5-flash",
    temperature=0.7,
    convert_system_message_to_human=True
    )

    
    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain



def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    
    response = chain(
        {"input_documents":docs, "question": user_question}
        , return_only_outputs=True)

    st.markdown(f"<span style='color:white'>{response}</span>", unsafe_allow_html=True)
    #st.write("Reply: ", response["output_text"])
    st.markdown(
    f"<p style='color:white;'>Reply: {response['output_text']}</p>",
    unsafe_allow_html=True
    )

def main():

    import asyncio

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)


    st.markdown(f"""
    <style>
    .stApp {{
        background-image: url("https://i.pinimg.com/originals/7f/e7/74/7fe774bbacce09f00cb5b2e3cbc48db3.png");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
    """, unsafe_allow_html=True)
    if 'page' not in st.session_state:
        st.session_state.page = 1

    def next_page():
        st.session_state.page += 1
    def prev_page():
        st.session_state.page -= 1
    def page1():
        st.markdown("<h1 style='color: white;'>UPLOAD</h1>", unsafe_allow_html=True)
        st.markdown("""
        <p style='color: white; font-size: 18px;'>
            Upload your PDF Files and Click on the Submit Button
        </p>
        """, unsafe_allow_html=True)
        pdf_docs = st.file_uploader(" ", accept_multiple_files=True)
        if st.button("Submit"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")
        st.button("Next", on_click=next_page)

    def page2():
        st.markdown("<h1 style='color: white;'>Chat PDF</h1>", unsafe_allow_html=True)
        st.markdown("""
        <p style='color: white; font-size: 18px;'>
            Ask questions from your pdf's!
        </p>
        """, unsafe_allow_html=True)
        user_question = st.text_input("Ask a Question from the PDF Files")

        if user_question:
            user_input(user_question)
       
        st.button("Back", on_click=prev_page)
        
    
    if st.session_state.page == 1:
        page1()
    elif st.session_state.page == 2:
        page2()

if __name__ == "__main__":
    main()

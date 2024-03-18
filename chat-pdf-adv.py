import os
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA, LLMChain
from langchain.prompts import PromptTemplate
import docx

def read_pdf(file):
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def read_doc(file):
    doc = docx.Document(file)
    text = ""
    for para in doc.paragraphs:
        text += para.text + "\n"
    return text

def read_txt(file):
    with file as f:
        text = f.read()
    return text

def main():
    load_dotenv()
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    st.set_page_config(page_title="Chat With Your Doc")
    #st.set_option("server.maxUploadSize", 100 * 1024 * 1024)  # 100MB

    supported_file_types = ["pdf", "doc", "docx", "txt"]

    st.markdown("""
        <style>
        .stApp {
            background-color: #F5F5F5;
            color: #333333;
            font-family: 'Helvetica Neue', sans-serif;
        }
        .stHeader {
            text-align: center;
            padding: 20px;
        }
        .stTextInput {
            width: 100%;
            padding: 10px;
            font-size: 16px;
            border-radius: 5px;
            border: 1px solid #CCCCCC;
        }
        .stButton {
            background-color: #4CAF50;
            color: black;
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
        }
        .stButton:hover {
            background-color: #3e8e41;
        }
        </style>
    """, unsafe_allow_html=True)

    st.header("Chat With Your Doc 💬")
    st.write("Upload a PDF, doc, or txt document and ask questions about its content. The app will provide detailed, accurate, and helpful responses based on the information in the file.")

    file = st.file_uploader("Upload your file", type=supported_file_types)

    if file is not None:
        if file.type.split("/")[-1] not in supported_file_types:
            st.error(f"Unsupported file type: {file.type}. Please upload a PDF, doc, txt.")
            st.stop()

        text = ""
        if file.type == "application/pdf":
            text = read_pdf(file)
        elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document" or file.type == "application/msword":
            text = read_doc(file)
        elif file.type == "text/plain":
            text = read_txt(file)

        if text:
            llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=os.getenv("GOOGLE_API_KEY"), convert_system_message_to_human=True)

            # Generate summary
            #summary_prompt = PromptTemplate(
            #    input_variables=["text"],
            #    template="Summarize the following text in a concise paragraph:\n\n{text}\n\nSummary:",
            #)

            #summary_chain = LLMChain(llm=llm, prompt=summary_prompt)
            #summary = summary_chain.run(text=text)

            # Generate example questions
            #question_generator_prompt = PromptTemplate(
            #    input_variables=["summary"],
            #    template="Based on the following summary of a document, generate 3 exploratory example questions related to the content:\n\nSummary: {summary}\n\nQuestions:\n1.",
            #)

            #question_generator_chain = LLMChain(llm=llm, prompt=question_generator_prompt)
            #example_questions = question_generator_chain.run(summary=summary)

            #st.write("Summary:")
            #st.write(summary)
            #st.write("Relavant questions based on the document summary:")
            #st.write(example_questions)

            text_splitter = CharacterTextSplitter(chunk_size=10000, chunk_overlap=1000, length_function=len)
            chunks = text_splitter.split_text(text)

            embeddings = GoogleGenerativeAIEmbeddings(google_api_key=os.getenv("GOOGLE_API_KEY"), model="models/embedding-001")
            vector_index = Chroma.from_texts(chunks, embeddings).as_retriever(search_kwargs={"k": 5})

            prompt_template = """
            You are an AI assistant tasked with providing users with detailed, accurate, and helpful responses based on the provided context. Your goal is to thoroughly understand the user's question and provide a comprehensive answer using the relevant information from the context.

            Context:

            {context}

            Question: {question}

            Response: To provide a detailed, accurate, and helpful response, consider the following:

            - Carefully read and understand the context to identify the most relevant information.
            - Thoroughly analyze the user's question to determine the specific information they are seeking.
            - Provide a clear and concise answer that directly addresses the question.
            - Support your answer with relevant details, examples, or explanations from the context.
            - If the context does not contain enough information to fully answer the question, acknowledge the limitations and provide any relevant insights you can offer.
            - Maintain a polite, professional, and helpful tone throughout your response.

            Your Response:

            """

            PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

            qa_chain = RetrievalQA.from_chain_type(
                llm,
                retriever=vector_index,
                return_source_documents=True,
                chain_type_kwargs={"prompt": PROMPT}
            )

            user_question = st.text_input("Ask a question about your document:", placeholder="E.g. What is the main topic of this document?")
            generate_button = st.button("Submit")

            if generate_button:
                if not user_question:
                    st.warning("Please enter a question to generate a response.")
                else:
                    with st.spinner("Generating response..."):
                        result = qa_chain({"query": user_question})
                        st.success("Answer")
                        if "I don't know" in result["result"]:
                            st.warning("The provided context does not contain enough information to answer your question. Please try rephrasing or asking a different question.")
                        st.write(result["result"])

if __name__ == "__main__":
    main()
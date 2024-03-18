# Chat With Your Doc
# A test project for ProtonLabs
# WebApp deployed on URL: 

## Overview
The "Chat With Your Doc" project is a web application designed to allow users to upload PDF, DOC, or TXT documents and ask questions about their content. The application utilizes various natural language processing (NLP) techniques and tools to provide detailed and accurate responses based on the information in the uploaded documents.

## Tools Used
- **Python**: The primary programming language used for development.
- **Streamlit**: A Python library used for creating interactive web applications.
- **Langchain**: A library for NLP tasks such as text splitting and question answering.
- **Chromadb**: A vector database used for storing and retrieving text embeddings.
- **Google Gemini-Pro**: A language model used for generating responses to user queries.
- **PyPDF2**: A Python library for reading PDF files.
- **python-docx**: A Python library for reading DOCX files.

## Features
- **File Upload**: Users can upload PDF, DOC, or TXT documents through the web interface.
- **Text Extraction**: The application extracts text content from the uploaded documents using appropriate libraries based on the file type.
- **Question Answering**: Users can ask questions about the document content, and the application provides detailed responses generated by Google Gemini-Pro.
- **Contextual Understanding**: The application employs Langchain's NLP capabilities to understand the context of user questions and provide relevant responses.
- **Interactive Interface**: The Streamlit web application provides a user-friendly interface for uploading documents and interacting with the chatbot.

## Running the Application
1. Ensure you have Python installed on your system.
2. Install the required dependencies listed in the `requirements.txt` file using `pip install -r requirements.txt`.
3. Set up environment variables including the Google API key.
4. Run the Python script containing the Streamlit application (`streamlit run chat-pdv-adv.py`).
5. Access the web application through the provided URL which is often (http://localhost:8501)

## Supported File Types
The application supports the following file types for document upload:
- PDF (.pdf)
- Microsoft Word (.doc, .docx)
- Plain Text (.txt)

## Future Improvements
- Enhance the question-answering model for more accurate and informative responses.
- Implement summarization and example question generation features for uploaded documents.
- Improve the user interface for better usability and aesthetics.
- Optimize document processing and retrieval for larger files.

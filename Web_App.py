__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import os
import openai
import shutil
import hashlib
from glob import glob
from datetime import datetime

from dotenv import load_dotenv

import nltk
nltk.download('punkt')

#import chromadb
import streamlit as st 
from langchain.document_loaders import DirectoryLoader
#from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
#from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma 
#from langchain_community.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI

#from langchain_community.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from PIL import Image

image = Image.open("img.png")
col1, col2 = st.columns([1,2])
with col1: 
     st.image(image,use_column_width=True)
#st.image(image)
with col2:
     st.title('ChatMate')
st.markdown("✨Greetings! I'm Capalytics, your AI tool for real-time insights into CAPA (Corrective and Preventive Actions) compliance within your organization!✨")

openai.api_key = st.secrets['OPENAI_API_KEY']

os.environ['OPENAI_API_KEY']=openai.api_key
temp_directory = 'tmp'

UPLOAD_NEW = 'Upload New File'
ALREADY_UPLOADED = 'Already uploaded'

chatmate_started = False

def load_docs(directory:str):
     """
     Loading documents from the given directory
    
     """
     loader = DirectoryLoader(directory)
     documents = loader.load()
     return documents

def split_docs(documents, chunk_size=10000, chunk_overlap=20):
     """
#     Splits the docs into chunks
    
#     """
     text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
     docs = text_splitter.split_documents(documents)
     return docs

#@st.cache_resource

def startup_event(last_update:str):
     """
     Loads all the necessary models & data once the server starts
    
     """
     print(f"{last_update}")
     directory = 'tmp/'
     documents = load_docs(directory)
     docs = split_docs(documents)
    
     embeddings = SentenceTransformerEmbeddings(model_name='all-MiniLM-L6-v2')
     persist_directory = "chroma_db"
    
     vectordb = Chroma.from_documents(
                       documents = docs,
                       embedding = embeddings,
                       persist_directory=persist_directory
                      
     )
    
     vectordb.persist()
    
     model_name = "gpt-3.5-turbo"
     llm = ChatOpenAI(model_name=model_name)
    
     db = Chroma.from_documents(docs, embeddings)
     chain = load_qa_chain(llm, chain_type="stuff", verbose=True)
     return db, chain

def get_answer(query:str,db,chain):
     """ 
     Queries the model with given question & returns the answer
     """
     prompt_instructions = """ You are a Q&A assistant named Vasuki. If anyone asks your name, remember to say your name is Vasuki. Please provide a detailed and thorough response to the following query. Ensure that the answer is clear, concise, and includes examples where appropriate. For all other inquiries, your main goal is to provide answers as accurately as possible, based on the instructions and context you have been given. If a question does not match the provided context or is outside the scope of the document, kindly advise the user to ask questions within the context of the document. If the user asks for a summary of the attached document, provide a detailed summary of the uploaded document. """
     modified_query = f"{prompt_instructions}\n{query}"
     
     matching_docs_score = db.similarity_search_with_score(modified_query)
     matching_docs = [doc for doc, score in matching_docs_score]
     answer = chain.run(input_documents=matching_docs, question=modified_query)
    
     ## Preparing the sources ##
     sources = [{
               'content': doc.page_content, 
               'metadata': doc.metadata,
               'score': score              
              
     } for doc, score in matching_docs_score]
     return {'answer': answer,
             'sources': sources}
    
def start_chatmate():
     db, chain = startup_event(last_db_updated)
     if 'openai_model' not in st.session_state:
         st.session_state['openai_model']='gpt-3.5-turbo'
     if 'messages' not in st.session_state:
         st.session_state.messages=[]
     for message in st.session_state.messages:
         with st.chat_message(message['role']):
             st.markdown(message['content'])
     if prompt := st.chat_input('hi ! how is it going ?'):
         st.session_state.messages.append({'role':'user','content':prompt})
         with st.chat_message('user'):
             st.markdown(prompt)
         with st.chat_message('assistant'):
             message_placeholder = st.empty()
             full_response = get_answer(st.session_state.messages[-1]['content'],db,chain)
             answer = full_response['answer']
             message_placeholder.markdown(answer)
             st.session_state.messages.append({'role':'assistant', 'content':answer})

content_type = st.sidebar.radio("Which knowledge base you want to use ?", [ALREADY_UPLOADED, UPLOAD_NEW])

if content_type == UPLOAD_NEW:
     uploaded_files = st.sidebar.file_uploader("choose a text file", accept_multiple_files=True)
     uploaded_file_names = [file.name for file in uploaded_files]
     if uploaded_files is not None and len(uploaded_files):
         if os.path.exists(temp_directory):
             shutil.rmtree(temp_directory)
         os.makedirs(temp_directory)
         if len(uploaded_files):
             last_db_updated = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
         for file in uploaded_files:
             with open(f"{temp_directory}/{file.name}", 'wb') as temp:
                 temp.write(file.getvalue())
                 temp.seek(0)
        
curr_dir = [path.split(os.path.sep)[-1] for path in glob(temp_directory + '/*')]

if content_type == ALREADY_UPLOADED:
     st.sidebar.write("Using Current Knowledge Base")
     if len(curr_dir):
         st.sidebar.write(curr_dir)
     else :
         st.sidebar.write("**No Knowledge Base Uploaded**")
    
last_db_updated = hashlib.md5(','.join(curr_dir).encode()).hexdigest()

if curr_dir and len(curr_dir):
     start_chatmate()
else :
     st.markdown("⚠️ No Knowledge Base Loaded, Please use the left menu to start ❗")

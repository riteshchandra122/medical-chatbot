from langchain_community.document_loaders import PyPDFLoader,DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
 #1) Load raw PDF files
DATA_PATH="data/"
def load_pdf_files(data):
    loader=DirectoryLoader(data,glob="**/*.pdf",loader_cls=PyPDFLoader)
    documents=loader.load()
    return documents

documents=load_pdf_files(data=DATA_PATH)
#print("Length of PDF pages:",len(documents))
#2) Create chunks of data
def create_chunks(extracted_data):
    text_splitter=RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50)
    docs=text_splitter.split_documents(extracted_data)
    return docs
text_chunks=create_chunks(extracted_data=documents)
#print("Length of text chunks:",len(text_chunks))

#3)vector embedding
def get_embedding_model():
    embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embedding_model
embedding_model=get_embedding_model()

#4) Create vector store
DB_FAISS_PATH="vector_store/db_faiss"
bd=FAISS.from_documents(text_chunks,embedding_model)
bd.save_local(DB_FAISS_PATH)

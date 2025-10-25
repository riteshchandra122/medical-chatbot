import os
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

#1)Setup LLM (Mistralwith HuggingFace)
HF_TOKEN=os.getenv("HF_TOKEN")
huggingface_repo_id="mistralai/Mistral-7B-Instruct-v0.3"
def load_llm(huggingface_repo_id):
    llm=HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        task="text-generation",
        model_kwargs={"temperature":0.5,"max_new_tokens":512},
        huggingfacehub_api_token=HF_TOKEN
    )
    return llm

#2) Connect LLM with FAISS and Create Chain

custom_prompt_template="""
Use the pieces of information provided in the context to answer user's question.
If you dont know the answer, just say that you dont know, dont try to make up an answer. 
Dont provide anything out of the given context

Context: {context}
Question: {question}

Start the answer directly. No small talk please.
"""

def set_custom_prompt(custom_prompt_template):
    prompt_template=PromptTemplate(
        input_variables=["context","question"],
        template=custom_prompt_template
    )
    return prompt_template

# Load Database
DB_FAISS_PATH="vectorstore/db_faiss"
embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db=FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# Create QA chain
qa_chain=RetrievalQA.from_chain_type(
    llm=load_llm(huggingface_repo_id),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={'k':3}),
    return_source_documents=True,
    chain_type_kwargs={'prompt':set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
)

# Now invoke with a single query
user_query=input("Write Query Here: ")
response=qa_chain.invoke({'query': user_query})
print("RESULT: ", response["result"])
print("SOURCE DOCUMENTS: ", response["source_documents"])
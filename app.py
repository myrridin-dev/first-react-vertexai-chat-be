from typing import List

import json
import os
import vertexai

from flask import Flask, request, jsonify
from flask_cors import CORS

from langchain import globals

from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import TextLoader, UnstructuredPDFLoader
from langchain_community.vectorstores import FAISS

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

from langchain.prompts import PromptTemplate
from langchain.schema.messages import HumanMessage
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_vertexai import VertexAI, VertexAIEmbeddings

PROJECT_ID='first-react-vertexai-chat-be'
REGION_ID='us-central1'

TEXT_EMBEDDING_MODEL = 'textembedding-gecko'
LLM_MODEL = "gemini-1.0-pro"

INDEX_PATH = './index/'
DB_PATH = '/tmp/'

ROLE_AI="ai"
ROLE_HUMAN="human"
ROLE_SYSTEM="system"

DEBUG = True

def get_split_documents(index_path: str) -> List[str]:
    chunk_size=1024
    chunk_overlap=128

    split_docs = []

    for file_name in os.listdir(index_path):
        if file_name.endswith(".pdf"):
            loader = UnstructuredPDFLoader(index_path + file_name)
        else:
            loader = TextLoader(index_path + file_name)

        text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        split_docs.extend(text_splitter.split_documents(loader.load()))

    return split_docs


def create_vector_db():
    embeddings = VertexAIEmbeddings(
        model_name=TEXT_EMBEDDING_MODEL, batch_size=5
    )
    # Load documents, generate vectors and store in Vector database
    split_docs = get_split_documents(INDEX_PATH)
    faissdb = FAISS.from_documents(split_docs, embeddings)
    faissdb.save_local(DB_PATH + '/faiss.db')

    return faissdb

def get_llm() -> VertexAI:
    return VertexAI(
        model='gemini-1.0-pro',
        max_output_tokens=8192,
        temperature=0.2,
        top_p=0.8,
        top_k=1,
        verbose=DEBUG,
    )

def get_embeddings() -> VertexAIEmbeddings:
    return VertexAIEmbeddings(
        model_name='textembedding-gecko',
        batch_size=5
    )

def get_system_prompt_template():
    return """
        You are a helpful AI assistant. You're tasked to answer the question given below, but only based on the context provided.

        The Tax Guide for Seniors provides a general overview of selected topics that are of interest to older tax-payers...

        Q: How do I report the amounts I set aside for my IRA?
        A: See Individual Retirement Arrangement Contributions and Deductions in chapter 3.

        Q: What are some of the credits I can claim to reduce my tax?
        A: See chapter 5 for discussions on the credit for the elderly or the disabled, the child and dependent care credit, and the earned income credit.

        Q: Must I report the sale of my home? If I had a gain, is any part of it taxable?
        A: See Sale of Home in chapter 2.

        context:
        <context>
        {context}
        </context>

        question:
        <question>
        {input}
        </question>

        If you cannot find an answer ask the user to rephrase the question.
        answer:
    """

# main
globals.set_debug(DEBUG)

vertexai.init(project=PROJECT_ID, location=REGION_ID)

llm = VertexAI(
    model=LLM_MODEL,
    max_output_tokens=8192,
    temperature=0.2,
    top_p=0.8,
    top_k=1,
    verbose=DEBUG,
)

faissdb = create_vector_db()
retriever = faissdb.as_retriever()

### Sub chain for contextualizing the question
# Goal: takes historical messages and the latest user question, and reformulates the question if it makes reference to any information in the historical information.
contextualize_q_system_prompt = """
    Given a chat history {history} and the latest user question {input} which might reference context in the chat history, \
    formulate a standalone question which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is.
"""
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        (ROLE_SYSTEM, contextualize_q_system_prompt),
        MessagesPlaceholder("history"),
        (ROLE_HUMAN, "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

### Main chain for chat with history
system_prompt = get_system_prompt_template()
qa_prompt = ChatPromptTemplate.from_messages(
    [
        (ROLE_SYSTEM, system_prompt),
        MessagesPlaceholder("history"),
        (ROLE_HUMAN, "{input}"),
    ]
)

qa_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

# Set up memory
msgs = ChatMessageHistory(key="langchain_messages") # The key to use in session state for storing messages.
if len(msgs.messages) == 0:
    msgs.add_ai_message("How can I help you?")

rag_chain_with_history = RunnableWithMessageHistory(
    rag_chain,
    lambda session_id: msgs,
    input_messages_key="input",
    output_messages_key="answer",   # This is what the response key in the vertexai response is
    history_messages_key="history", # This is specifying the key where Langchain automatically stores new messages as history?
)

# Flask section

app = Flask(__name__)
CORS(app)

@app.route('/predict', methods= ['POST'])
def predict():
    if request.get_json():
        x=json.dumps(request.get_json())
        print('ok')
        x=json.loads(x)
    else:
        x={}
    data=x["text"]  # text

    # This ensures new messages are saved to history automatically by Langchain during chain run
    config = {"configurable": {"session_id": "any"}}
    
    response = rag_chain_with_history.invoke({"input": data}, config)
    if DEBUG:
        print(f"[FULL] ai response: {response}")
    answer = response["answer"]
    source_documents = list({doc.metadata["source"] for doc in response["context"]})

    response = {
        'answer': answer,
        'sources': source_documents
    }

    response=jsonify(response)
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

# TODO
# /index will take in a file and call create_vector_db(file)
# @app.route('/index', methods= ['POST'])

if __name__ == "__main__":
    app.run(port=8080, host='0.0.0.0', debug=DEBUG)
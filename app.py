from flask import Flask, render_template, jsonify, request
from src.helper import (
    download_hugging_face_embeddings,
    detect_language,
    translate_to_english,
    translate_from_english
)
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os


# ----------------------------------------------------
# Flask App Initialization
# ----------------------------------------------------
app = Flask(__name__)


load_dotenv()

# ----------------------------------------------------
# Environment Variables
# ----------------------------------------------------
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GEMINI_AI_API_KEY = os.getenv("GEMINI_AI_API_KEY") 

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GEMINI_AI_API_KEY"] = GEMINI_AI_API_KEY

# ----------------------------------------------------
# Multilingual Embeddings (mBERT)
# ----------------------------------------------------

embeddings = download_hugging_face_embeddings()


# ----------------------------------------------------
# Pinecone Index Connection
# ----------------------------------------------------

index_name = "medical-chatbot" 
# Embed each chunk and upsert the embeddings into your Pinecone index.
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)


retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})

# ----------------------------------------------------
# Chat Model (Gemini)
# ----------------------------------------------------
chatModel = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",  # or "gemini-1.5-pro"
    google_api_key="AIzaSyCD9WXurWfnPEftf7IwMGMePbQxs1h11nk"
)

os.environ.pop("GOOGLE_APPLICATION_CREDENTIALS", None)
os.environ["GOOGLE_API_KEY"] = "AIzaSyCD9WXurWfnPEftf7IwMGMePbQxs1h11nk"

chatModel = ChatGoogleGenerativeAI(model="gemini-2.5-flash")  # or "gemini-1.5-pro"


# ----------------------------------------------------
# Prompt + RAG Chain
# ----------------------------------------------------

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(chatModel, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)


# ----------------------------------------------------
# Routes
# ----------------------------------------------------

@app.route("/")
def index():
    return render_template('chat.html')



@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    print(f"User Query: {msg}")

    # 1️⃣ Detect user input language
    user_lang = detect_language(msg)
    print(f"Detected language: {user_lang}")

    # 2️⃣ Translate user query to English (if needed)
    english_query = translate_to_english(msg)
    print(f"Translated to English: {english_query}")

    # 3️⃣ Get English response from RAG chain
    response = rag_chain.invoke({"input": english_query})
    english_answer = response["answer"]
    print(f"English Answer: {english_answer}")

    # 4️⃣ Translate response back to user’s language
    translated_answer = translate_from_english(english_answer, user_lang)
    print(f"Translated Answer: {translated_answer}")

    # 5️⃣ Return translated response to frontend
    return str(translated_answer)

# ----------------------------------------------------
# Run Server
# ----------------------------------------------------

if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)
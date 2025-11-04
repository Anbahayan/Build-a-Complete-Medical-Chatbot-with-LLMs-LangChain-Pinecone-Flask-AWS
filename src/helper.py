from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from typing import List
from langchain.schema import Document
from deep_translator import GoogleTranslator
from langdetect import detect

#Extract Data From the PDF File
def load_pdf_file(data):
    loader= DirectoryLoader(data,
                            glob="*.pdf",
                            loader_cls=PyPDFLoader)

    documents=loader.load()

    return documents



def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    """
    Given a list of Document objects, return a new list of Document objects
    containing only 'source' in metadata and the original page_content.
    """
    minimal_docs: List[Document] = []
    for doc in docs:
        src = doc.metadata.get("source")
        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata={"source": src}
            )
        )
    return minimal_docs



#Split the Data into Text Chunks
def text_split(extracted_data):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    text_chunks=text_splitter.split_documents(extracted_data)
    return text_chunks



"""#Download the Embeddings from HuggingFace 
def download_hugging_face_embeddings():
    embeddings=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')  #this model return 384 dimensions
    return embeddings"""

# ----------------------------------------------------
# 4️⃣ Multilingual Embedding Model (mBERT)
# ----------------------------------------------------

def download_hugging_face_embeddings():
    """
    Uses multilingual BERT for language-independent embeddings.
    Supports 100+ languages.
    """
    embeddings = HuggingFaceEmbeddings(
        model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
    )
    return embeddings


# ----------------------------------------------------
# 5️⃣ Translation & Language Detection
# ----------------------------------------------------
def detect_language(text: str) -> str:
    """Detect user input language (e.g., 'en', 'hi', 'ta', 'es', etc.)."""
    try:
        lang = detect(text)
        return lang
    except Exception:
        return "en"

def translate_to_english(text: str) -> str:
    """Translate any language → English."""
    try:
        return GoogleTranslator(source='auto', target='en').translate(text)
    except Exception:
        return text

def translate_from_english(text: str, target_lang: str) -> str:
    """Translate English → user's language."""
    try:
        if target_lang == "en":
            return text
        return GoogleTranslator(source='en', target=target_lang).translate(text)
    except Exception:
        return text
import os
from openai import OpenAI
from dotenv import load_dotenv
from pypdf import PdfReader
import umap
import numpy as np

from helper_utils import word_wrap

# Load environment variables from .env file
load_dotenv()

openai_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_key) 

reader = PdfReader("data/microsoft-annual-report.pdf")
pdf_texts = [p.extract_text().strip() for p in reader.pages]

# Filter the empty strings
pdf_texts = [text for text in pdf_texts if text]

from langchain.text_splitter import (
    RecursiveCharacterTextSplitter, 
    SentenceTransformersTokenTextSplitter,  
)

character_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", ". ", " ", ""], chunk_size=1000,chunk_overlap=0)
character_split_texts = character_splitter.split_text("\n\n".join(pdf_texts))

token_splitter = SentenceTransformersTokenTextSplitter(tokens_per_chunk=256, chunk_overlap=0)

token_split_texts = []
for text in character_split_texts:
    token_split_texts += token_splitter.split_text(text)
    
# now we import chromadb and the SentenceTransformerEmbeddingFunction to create an embedding function that will be used to create embeddings for our text chunks.
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

embedding_function = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
# print(embedding_function(token_split_texts[10]))

# we then instantiate the Chroma client and create a collection to store our embeddings. 
chroma_client = chromadb.Client()
chroma_collection = chroma_client.create_collection(name="microsoft-annual-report", embedding_function=embedding_function)


# extract the embeddings of the token_split_texts
ids = [str(i) for i in range(len(token_split_texts))]

chroma_collection.add(ids=ids, documents=token_split_texts,)
chroma_collection.count()

query = "What was the tota revenue for the year?"

results = chroma_collection.query(query_texts=[query], n_results=5)
retrieved_documents = results["documents"][0]

for document in retrieved_documents:
    print(word_wrap(document))
    print("\n---\n")         




































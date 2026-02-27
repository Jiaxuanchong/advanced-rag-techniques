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

query = "What was the total revenue for the year?"

results = chroma_collection.query(query_texts=[query], n_results=5)
retrieved_documents = results["documents"][0]

# for document in retrieved_documents:
#     print(word_wrap(document))
#     print("\n---\n")         

def generate_multi_query(query, model="gpt-3.5-turbo"):
    
    prompt = """You are a knowledgeable financial research assistant. 
    Your users are inquiring about an annual report. For the given question,
    propose up to five related questions to assist them in finding the information they need.
    Provide concise, single-topic questions (without compounding sentences) that cover various aspects of the topic.
    Ensure each question is complete and directly related to the original inquiry.
    List each question on a seperate line without numbering. 
    """
    
    messages = [
        {
            "role": "system",
            "content": prompt
        },
        {
            "role": "user",
            "content": query
        }
    ]
    
    response = client.chat.completions.create(
        model = model,
        messages = messages,
    )
    content = response.choices[0].message.content
    content = content.split("\n")
    return content


original_query = ("What details can you provide about the factors that led to revenue growth?")

aug_queries = generate_multi_query(original_query)

# 1. First step show the augmented queries
for query in aug_queries:
    print("\n", query)
    
# 2. concatentate the orignal query with the augmented queries
joint_query = [original_query] + aug_queries
# original query is in a list becuase chroma can actually handle multiple queries, so we add it in a list

# print("======> \n\n", joint_query)

results = chroma_collection.query(
    query_texts=joint_query, n_results=5, include=["documents", "distances"]
)

retrieved_documents = results["documents"]

# Deduplicate the retrieved documents
unique_documents = set()
for documents in retrieved_documents:
    for document in documents:
        unique_documents.add(document)
        
# output the results documents
for i, documents in enumerate(retrieved_documents):
    print(f"Query: {joint_query[i]}")
    print("")
    print("Results:")
    for doc in documents:
        print(word_wrap(doc))
        print("")
    print("-" * 100)
    
    
# Deduplicated documents are stored in unique_documents

context = "\n\n".join(unique_documents)

print("\n================== FINAL CONTEXT ==================\n")
print(word_wrap(context[:2000]))  # print only first 2000 chars to avoid flooding


def generate_answer(query, context, model="gpt-4o-mini"):
    
    prompt = f"""
You are a financial analyst assistant.

Answer the question based ONLY on the provided context.
If the answer is not contained in the context, say:
"The information is not available in the provided document."

Be precise and structured in your explanation.

Context:
{context}

Question:
{query}
"""
    
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a professional financial analyst."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )
    
    return response.choices[0].message.content


final_answer = generate_answer(original_query, context)

print("\n================== FINAL ANSWER ==================\n")
print(word_wrap(final_answer))

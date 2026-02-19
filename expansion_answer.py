from helper_utils import project_embeddings, word_wrap
from pypdf import PdfReader
from openai import OpenAI
from dotenv import load_dotenv
import os

from pypdf import PdfReader
import umap

# Load environment variables from .env file
load_dotenv()

openai_key = os.getenv("OPENAI_API_KEY") 
client = OpenAI(api_key=openai_key)

reader = PdfReader("data/microsoft-annual-report.pdf")

# extract text from the pages
pdf_texts = [p.extract_text().strip() for p in reader.pages]

#Filter the empty strings
pdf_texts = [text for text in pdf_texts if text]
# print(
#     word_wrap(
#         pdf_texts[0],
#         width=100,
#     )
# )

# Split the text into smaller chunks

from langchain.text_splitter import (
    RecursiveCharacterTextSplitter, 
    SentenceTransformersTokenTextSplitter,
)

# RecursiveCharacterTextSplitter splits long text into smaller chunks, using characters as measurement.
# SentenceTransformersTokenTextSplitter splits long text into smaller chunks, using tokens as measurement. 

character_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", ". ", " ", ""], chunk_size=1000,chunk_overlap=0)

character_split_texts = character_splitter.split_text("\n\n".join(pdf_texts))

# print(word_wrap(character_split_texts[10]))
# print(f"\nTotal chunks: {len(character_split_texts)}")

token_splitter = SentenceTransformersTokenTextSplitter(
    tokens_per_chunk=256,
    chunk_overlap=32
)
token_split_texts = []
for text in character_split_texts:
    token_split_texts += token_splitter.split_text(text)
    
# print(word_wrap(token_split_texts[10]))
# print(f"\nTotal chunks: {len(token_split_texts)}")


import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

embedding_function = SentenceTransformerEmbeddingFunction()
# print(embedding_function([token_split_texts[10]]))

chroma_client = chromadb.Client()
chroma_collection = chroma_client.create_collection(name="microsoft-annual-report", embedding_function=embedding_function)

# extract the embeddings of the token_split_texts
ids = [str(i) for i in range(len(token_split_texts))]

chroma_collection.add(
    ids=ids,
    documents=token_split_texts,
)

chroma_collection.count()

query = "What was the total revenue for the year?"

results = chroma_collection.query(
    query_texts=[query],
    n_results=5,
)       

retrieved_docouments = results['documents'][0]

for document in retrieved_docouments:
    print(word_wrap(document))
    print("\n")
    
    
def augment_query_generated(query, model="gpt-3.5-turbo"):
    client = OpenAI(api_key=openai_key)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful expert financial research assistant. Provide an example answer to the given question, that might be found in a document like an annual report."
            },
            {
                "role": "user",
                "content": query
            }
        ]
    )

    content = response.choices[0].message.content
    return content


original_query = "What was the total profit for the year, and how does it compare to the previous year?"
hypothetical_answer = augment_query_generated(original_query)

joint_query = f"Question: {original_query}{hypothetical_answer}"
print(word_wrap(joint_query))  

results = chroma_collection.query(
    query_texts=joint_query,
    n_results=5,
    include=["documents", "embeddings"]
) 

retrieved_docouments = results['documents'][0]

# for doc in retrieved_docouments:
#     print(word_wrap(doc))
#     print("\n")

embeddings = chroma_collection.get(include=["embeddings"])["embeddings"]
umap_transform = umap.UMAP(random_state=0, transform_seed=0).fit(embeddings)
projected_dataset_embeddings = project_embeddings(embeddings, umap_transform)

retrieved_embedding = results['embeddings'][0]
original_query_embedding = embedding_function([original_query])
augmented_query_embedding = embedding_function([joint_query])

projected_original_query_embedding = project_embeddings(original_query_embedding, umap_transform)

projected_augmented_query_embedding = project_embeddings(augmented_query_embedding, umap_transform)

projected_retrieved_embeddings = project_embeddings(retrieved_embedding, umap_transform)

 
# Advanced RAG Techniques

## Project Overview
This repository contains example code and utilities for building Retrieval-Augmented Generation (RAG) pipelines focused on document-heavy workflows (PDFs, text files, etc.). It demonstrates parsing documents, expanding user queries to improve retrieval, and using a language model to generate context-aware answers grounded in retrieved content.

## Goals
- Provide clear, runnable examples of RAG components.
- Show practical query expansion techniques to improve retrieval recall.
- Offer utility functions for text processing and simple I/O around PDFs.

## Key Features
- PDF parsing and text extraction using `pypdf`.
- Query expansion helpers to broaden retrieval scope.
- Integration examples for sending context + queries to an LLM (OpenAI).
- Small utilities for formatting and presentation in `helper_utils.py`.

## Repository Layout
- `expansion_queries.py`: scripts and helpers for generating expanded queries.
- `expansion_answer.py`: example end-to-end flow: load docs, retrieve context, call the LLM, and format output.
- `helper_utils.py`: small utility functions (e.g., word wrapping, simple cleanup).
- `data/`: place sample PDFs or other documents used for testing.


## How It Works (high level) 

1. Load environment variables and initialize OpenAI client.
2. Read and extract text from PDF(s) in `data/` using `pypdf`.
3. Filter and clean extracted text.
4. Split text into smaller chunks using both character-based and token-based splitters from LangChain.
5. Generate embeddings for each chunk using SentenceTransformer.
6. Store chunks and embeddings in a ChromaDB collection.
7. Query the collection for relevant chunks using a user question.

### expansion_answer.py
8. Use OpenAI to generate multiple related queries (query expansion) for the original question.
9. Query ChromaDB with both the original and expanded queries to retrieve relevant documents.
10. Deduplicate retrieved documents and concatenate them for context.
11. Use OpenAI to generate a final answer based only on the retrieved context.
12. Print the final answer and context for review.

### expansion_queries.py 
8. Use OpenAI to generate multiple related queries (query expansion) for the original question.
9. Query ChromaDB with both the original and expanded queries to retrieve relevant documents.
10. Deduplicate retrieved documents and concatenate them for context.
11. Use OpenAI to generate a final answer based only on the retrieved context.
12. Print the final answer and context for review.


## Notes & Tips
- Keep documents reasonably sized or chunk them before embedding/retrieval to avoid truncation.
- Query expansion can improve recall but may increase retrieved noise — tune according to your data.
- This repo uses simple, illustrative examples rather than production-grade retrieval stacks. For production, consider vector databases (e.g., FAISS, Weaviate) and batching.



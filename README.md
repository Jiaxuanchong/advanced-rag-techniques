# Advanced RAG Techniques

## Introduction
This repository explores advanced retrieval-augmented generation (RAG) techniques for building intelligent document processing pipelines. It demonstrates how to combine large language models with custom document retrieval to generate contextually relevant answers from PDF sources.

Key features:
- PDF document parsing and extraction
- Integration with OpenAI's API for intelligent question-answering
- Query expansion techniques for improved retrieval
- Utility functions for text processing and formatting

Whether you're building a document Q&A system, knowledge base assistant, or experimenting with prompt engineering, this repo provides practical examples to get you started.

## What are Advanced RAG Techniques?

Retrieval-Augmented Generation (RAG) is a powerful approach that combines information retrieval with generative AI models to produce contextually relevant and accurate responses. Instead of relying solely on the model's pre-trained knowledge, RAG techniques retrieve external information (e.g., from documents, databases, or APIs) to enhance the quality of generated answers.

### Key Concepts in Advanced RAG Techniques:
1. **Document Retrieval**:
   - Extract relevant information from external sources, such as PDFs, databases, or web pages, to provide context for the AI model.
   - In this repository, the `PdfReader` library is used to parse and extract text from PDF documents.

2. **Query Expansion**:
   - Enhance user queries by adding related terms or rephrasing them to improve the retrieval of relevant information.
   - Query expansion ensures that the AI model has access to the most relevant context for generating accurate responses.

3. **Contextual Generation**:
   - Use the retrieved information as input to a generative AI model (e.g., OpenAI's GPT) to produce responses that are grounded in the provided context.
   - This approach reduces hallucinations and improves the factual accuracy of the generated content.

4. **Text Preprocessing**:
   - Preprocess the retrieved text to ensure it is clean, concise, and formatted for optimal input into the AI model.
   - The `helper_utils` module in this repository provides functions like `word_wrap` to format text for better readability.

### Benefits of Advanced RAG Techniques:
- Improved accuracy and relevance of AI-generated responses.
- Ability to handle domain-specific queries by leveraging external knowledge sources.
- Enhanced user experience through more precise and context-aware answers.

## How It Works

### Code Logic Overview
The main script, `expansion_answer.py`, demonstrates how to process PDF documents and use OpenAI's API for generating intelligent responses. Below is a breakdown of the key components:

1. **Environment Setup**:
   - The script uses the `dotenv` library to load environment variables from a `.env` file.
   - The `OPENAI_API_KEY` is retrieved from the environment to authenticate requests to the OpenAI API.

   ```python
   from dotenv import load_dotenv
   import os

   load_dotenv()
   openai_key = os.getenv("OPENAI_API_KEY")
   ```

2. **PDF Parsing**:
   - The `PdfReader` from the `pypdf` library is used to read and extract text from PDF files.
   - This allows the script to process documents and use their content for generating answers.

   ```python
   from pypdf import PdfReader
   # Example usage:
   # reader = PdfReader("example.pdf")
   # text = " ".join([page.extract_text() for page in reader.pages])
   ```

3. **Text Formatting**:
   - The `word_wrap` function from `helper_utils` is used to format the extracted text or generated responses for better readability.

   ```python
   from helper_utils import word_wrap
   # Example usage:
   # formatted_text = word_wrap(raw_text, width=80)
   ```

4. **OpenAI API Integration**:
   - The script uses the `OpenAI` library to interact with OpenAI's API.
   - Queries are sent to the API, and responses are generated based on the extracted PDF content and user input.


### Workflow
1. Load the `.env` file to retrieve the OpenAI API key.
2. Use `PdfReader` to extract text from the PDF file.
3. Format the extracted text using `word_wrap` for better readability.
4. Send the processed text and user query to the OpenAI API to generate a response.
5. Output the response in a user-friendly format.

## Contributing
- Open issues or PRs for improvements.

## License
- Add a license file as needed (e.g., MIT).

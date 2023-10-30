---
title: Search Offers
emoji: 🕵️
colorFrom: green
colorTo: yellow
sdk: streamlit
sdk_version: 1.28.0
app_file: app.py
pinned: false
---

# Project Title

## Description

This project aims to perform offer search using various Natural Language Processing (NLP) and information retrieval techniques. Different models and approaches have been experimented with to optimize the search results based on user queries.

## Technologies and Models

- **Spacy**: Used for basic NLP tasks such as tokenization and similarity calculations.
- **TF-IDF**: Implemented for keyword extraction and to calculate text relevance.
- **BERT (Hugging Face)**: Leveraged pre-trained models for generating rich text embeddings.
- **OpenAI Embeddings**: Utilized for creating embeddings and leveraging language models for similarity calculations.

## Approaches and Experimentations

- **Direct Matching**: Tried direct matching of query terms with offer details.
- **Fuzzy Matching**: Utilized `fuzzywuzzy` to add flexibility to text matching.
- **Vector Stores**: Experimented with various vector stores such as Quadrant and FAISS to optimize speed and performance.
- **Caching**: Experimented with caching the vector store for performance enhancement.
- **Embedding Caching**: Tried different embedding caching mechanisms to speed up the retrieval process.
- **Language Models**: Experimented with various language models including Langchain to enhance the semantic search capabilities.

## Deployment

The application is deployed using Hugging Face Spaces, integrating it seamlessly with the Streamlit application, allowing for a user-friendly interface and easy interaction.

## How to Use

1. Navigate to the deployed application.
2. Enter your search query.
3. Choose the search method.
4. Execute the search to view the matched offers.

## Conclusion and Findings

Through various experimentations, a combination of different models and approaches were tested to find the most optimal solution for offer searching, each with its own set of advantages and trade-offs.


Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

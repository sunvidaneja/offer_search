import streamlit as st
from utils.utils import (
    load_spacy_model,
    load_data,
    search_tfidf,
    metadata_entity_retrieval,
    document_embedding_retrieval,
    enhanced_search_spacy,
)
from langchain.embeddings.openai import OpenAIEmbeddings


class OfferSearch:
    """
    A class to execute a combination of NLP, LLMs and vector-based searches on a dataset using different methods.
    """

    def __init__(self):
        """
        Initializes the VectorDBSearch with necessary configurations.
        """
        self.api_key = st.secrets["OPENAI_API_KEY"]
        self.embeddings = OpenAIEmbeddings(openai_api_key=self.api_key)
        self.nlp = load_spacy_model()
        self.df = load_data()

    def execute_search(self, query, method):
        """
        Executes the search on the dataset based on the selected method.

        Parameters:
        - query: str
            The search query.
        - method: str
            The method used for searching.

        Returns:
        - DataFrame
            The search results.
        """
        if method == "Direct and TF-IDF Method":
            return search_tfidf(self.df, query)
        elif method == "Metadata Entity Filtering Search":
            return metadata_entity_retrieval(self.df, query, self.embeddings)
        elif method == "Document Embedding Retrieval Search (LLM)":
            return document_embedding_retrieval(self.df, query, self.embeddings)
        else:
            return enhanced_search_spacy(self.df, query)

    def main(self):
        """
        Main method to run the Streamlit application.
        Handles user inputs and displays the results.
        """
        st.title("Search Offers")

        query = st.text_input("Enter your query:")
        st.markdown(
            '_You can also try combinations like "Walmart Target" or "Walmart Cookies"._'
        )

        method = st.selectbox(
            "Select the method:",
            [
                "Spacy Search",
                "Direct and TF-IDF Method",
                "Metadata Entity Filtering Search",
                "Document Embedding Retrieval Search (LLM)",
            ],
        )

        if st.button("Execute"):
            results = self.execute_search(query, method)

            if results.empty:
                st.write("No results found.")
            else:
                st.write("Results:")
                st.dataframe(results)


if __name__ == "__main__":
    app = OfferSearch()
    app.main()

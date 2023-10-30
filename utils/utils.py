import os
import numpy as np
import pandas as pd
import re
from tqdm.notebook import tqdm
from langchain.schema import Document
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
from fuzzywuzzy import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import chromadb
from rouge import Rouge
import spacy
from spacy.cli.download import download as spacy_download
from constants import MODEL_NAME, DATA_PATH


def load_spacy_model():
    """
    Load the SpaCy model for natural language processing.

    Returns:
    - spacy.lang.en.English: Loaded SpaCy model.
    """
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        spacy_download("en_core_web_sm")
        return spacy.load("en_core_web_sm")


def load_data():
    """
    Load the data from a CSV file specified in the constants.

    Returns:
    - pandas.DataFrame: Loaded data in a pandas DataFrame.
    """
    return pd.read_csv(DATA_PATH)


def jaccard_similarity(str1, str2):
    """
    Calculate the Jaccard similarity between two strings.

    Parameters:
    - str1 (str): First string.
    - str2 (str): Second string.

    Returns:
    - float: Jaccard similarity score.
    """
    a = set(str1.split())
    b = set(str2.split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))


def calculate_rouge_score(query, text):
    """
    Calculate the ROUGE-L score between two texts.

    Parameters:
    - query (str): Query text.
    - text (str): Text to compare against the query.

    Returns:
    - float: ROUGE-L score.
    """
    rouge = Rouge()
    scores = rouge.get_scores(query, text)
    return scores[0]["rouge-l"]["f"]


def clean_text_all_special_chars(text):
    """
    Remove all special characters, punctuation, and symbols except alphanumeric and space
    from the input text.

    Parameters:
    - text : str
        The text to be cleaned.

    Returns:
    - str
        The cleaned text.
    """
    cleaned_text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    return cleaned_text


def metadata_entity_retrieval(df, query, embeddings, top_n=5):
    """
    This function operates on entity embeddings to find similar entities within a dataset.
    Entities from specified columns are embedded, and similarity scores are calculated against
    the query. The dataset is then filtered to keep rows containing the most relevant entities.

    Parameters:
    - df (pandas.DataFrame): The input data.
    - query (str): The user query to find similar entities.
    - top_n (int, optional): Number of top similar entities to retrieve. Default is 5.

    Returns:
    - pandas.DataFrame: Filtered DataFrame based on similarity search.
    """
    docs = []
    for c in ["RETAILER", "BRAND", "BRAND_CATEGORY"]:
        names = df[c].unique()
        docs += [
            Document(page_content=x.lower(), metadata={"Source": c})
            for x in names
            if not pd.isna(x)
        ]

    vectorstore1 = Chroma.from_documents(docs, embeddings, collection_name="metadata")
    res = vectorstore1.similarity_search_with_score(query, top_n)

    results = pd.DataFrame(columns=["Entity", "Source", "Score"])
    for i, (r, score) in enumerate(res):
        results.loc[i, "Entity"] = r.page_content
        results.loc[i, "Source"] = str(r.metadata["Source"])
        results.loc[i, "Score"] = 1 - score

        def filter_df(df1, df2):
            final_df = pd.DataFrame(columns=["OFFER", "Score"])
            for index, row in df1.iterrows():
                entity, source, score = row["Entity"], row["Source"], row["Score"]
                current_df = df2[df2[source].str.lower() == entity.lower()].copy()
                current_df = current_df[["OFFER"]]  # Keep only the 'OFFER' column
                current_df["Score"] = score  # Assigning score to the filtered rows
                final_df = pd.concat([final_df, current_df])

            return final_df.sort_values(by="Score", ascending=False).reset_index(
                drop=True
            )

    return filter_df(results, df)


def document_embedding_retrieval(df, query, embeddings):
    """
    Utilize document embeddings and a retriever to find relevant offers within a dataset.
    The method includes associated metadata such as retailer, brand, and category in the
    embeddings. A retriever then finds and presents the most relevant documents based on the query.

    Parameters:
    - df (pandas.DataFrame): The input data containing offers and associated metadata.
    - query (str): The user query to find similar offers.

    Returns:
    - pandas.DataFrame: DataFrame containing relevant documents based on the query, along with ROUGE scores.
    """
    gcp_client = chromadb.PersistentClient(path="vectorstore")
    docs2 = []

    for i in range(len(df)):
        content = f"OFFER: {df.OFFER[i]}"
        metadata = {}
        for c in ["RETAILER", "BRAND", "BRAND_CATEGORY", "PARENT_CATEGORY"]:
            if not pd.isna(df[c][i]):
                metadata[c] = df[c][i]
                content += f", {c}: {df[c][i]}"
        metadata["OFFER"] = df.OFFER[i]
        docs2.append(Document(page_content=content, metadata=metadata))

    vectorstore2 = Chroma(
        client=gcp_client, collection_name="offers1", embedding_function=embeddings
    )

    metadata_field_info = [
        AttributeInfo(
            name="RETAILER",
            description="Name of the retailer providing the offer",
            type="string",
        ),
        AttributeInfo(
            name="BRAND",
            description="Name of the brand for which the product belongs to",
            type="string",
        ),
        AttributeInfo(
            name="BRAND_CATEGORY",
            description="A category name the brand falls in",
            type="string",
        ),
        AttributeInfo(
            name="PARENT_CATEGORY",
            description="A higher level category for the brand category",
            type="string",
        ),
    ]

    document_content_description = (
        "Product offers belonging to different brands, products, and categories"
    )
    llm = OpenAI(temperature=0)
    retriever = SelfQueryRetriever.from_llm(
        llm,
        vectorstore2,
        document_content_description,
        metadata_field_info,
        verbose=True,
        search_kwargs={"k": 150},
    )

    res = retriever.get_relevant_documents(query.upper())

    results = pd.DataFrame(columns=["OFFER", "ROUGE_Score", "Jaccard_Score"])

    for i, r in enumerate(res):
        concatenated_text = f'{r.metadata["OFFER"]} {r.metadata.get("RETAILER", "")} {r.metadata.get("BRAND", "")} {r.metadata.get("BRAND_CATEGORY", "")} {r.metadata.get("PARENT_CATEGORY", "")}'

        rouge_score = calculate_rouge_score(query.upper(), concatenated_text)
        results.loc[i, "ROUGE_Score"] = rouge_score

        jaccard_score = jaccard_similarity(query.upper(), concatenated_text)
        results.loc[i, "Jaccard_Score"] = jaccard_score

        results.loc[i, "OFFER"] = r.metadata["OFFER"]

    results = results.sort_values(
        by=["ROUGE_Score", "Jaccard_Score"], ascending=False
    ).reset_index(drop=True)

    return results


def enhanced_search_spacy(df, query):
    """
    Perform an enhanced search on the DataFrame treating the entire query as a single entity.
    It calculates similarity scores between the query and concatenated text from the DataFrame.

    Parameters:
    - df : DataFrame
        The input DataFrame containing offers and their details.
    - query : str
        The query to find relevant documents.

    Returns:
    - DataFrame or None
        A DataFrame containing the top matches based on similarity scores, or None if no matches found.
    """
    results = pd.DataFrame()

    df["concatenated_text"] = df[
        ["OFFER", "BRAND", "BRAND_CATEGORY", "RETAILER"]
    ].apply(lambda row: " ".join(row.dropna()), axis=1)

    scores = df.apply(
        lambda row: fuzz.token_set_ratio(
            query.lower(), str(row["concatenated_text"]).lower()
        )
        / 100,
        axis=1,
    )
    df["similarity_score"] = scores

    top_matches = df.nlargest(100, "similarity_score")
    results = pd.concat([results, top_matches])

    results.reset_index(drop=True, inplace=True)

    if results.empty:
        print("No matching offers found.")
    else:
        return results[["OFFER", "similarity_score"]]


def search_tfidf(df, query, top_n=5):
    """
    Enhanced search function to find the most relevant offers based on the user's query.
    The function performs direct matching and calculates similarity scores based on TF-IDF vectorization
    and cosine similarity.

    Parameters:
    - df : DataFrame
        The input DataFrame containing offers and their details.
    - query : str
        The query to find relevant documents.
    - top_n : int, optional (default is 5)
        The number of top similar offers to retrieve.

    Returns:
    - DataFrame
        A DataFrame containing the top matches based on direct matches and similarity scores.
    """
    query = query.lower()

    direct_matches = df[
        (df["BRAND"].str.lower() == query)
        | (df["BRAND_CATEGORY"].str.lower() == query)
        | (df["PARENT_CATEGORY"].str.lower() == query)
        | (df["RETAILER"].str.lower() == query)
    ]

    direct_matches = direct_matches.copy()
    direct_matches["Score"] = 1

    df_remaining = df.drop(index=direct_matches.index)

    vectorizer = TfidfVectorizer(stop_words="english", lowercase=True)
    tfidf_matrix = vectorizer.fit_transform(df_remaining["OFFER"])
    query_vector = vectorizer.transform([query])

    cosine_similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    top_offers_indices = cosine_similarities.argsort()[-top_n:][::-1]

    similarity_matches = df_remaining.iloc[top_offers_indices].copy()
    similarity_matches["Score"] = cosine_similarities[top_offers_indices]

    all_matches = pd.concat([direct_matches, similarity_matches])

    all_matches = all_matches.dropna(subset=["Score"])
    all_matches = all_matches.sort_values(by="Score", ascending=False).reset_index(
        drop=True
    )

    return all_matches[["OFFER", "Score"]]

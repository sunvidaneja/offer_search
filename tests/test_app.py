import unittest
from utils import (
    load_spacy_model,
    load_data,
    search_tfidf,
    metadata_entity_retrieval,
    document_embedding_retrieval,
    enhanced_search_spacy,
)
from langchain.embeddings.openai import OpenAIEmbeddings


class TestSearchMethods(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.api_key = "YOUR_OPENAI_API_KEY"
        cls.embeddings = OpenAIEmbeddings(openai_api_key=cls.api_key)
        cls.nlp = load_spacy_model()
        cls.df = load_data()
        cls.query = "walmart"

    def test_search_tfidf(self):
        result = search_tfidf(self.df, self.query)
        self.assertIsNotNone(result)
        self.assertTrue("OFFER" in result.columns)
        self.assertTrue("Score" in result.columns)

    def test_enhanced_search_spacy(self):
        result = enhanced_search_spacy(self.df, self.query)
        self.assertIsNotNone(result)
        self.assertTrue("OFFER" in result.columns)
        self.assertTrue("similarity_score" in result.columns)

    def test_metadata_entity_retrieval(self):
        result = metadata_entity_retrieval(self.df, self.query, self.embeddings)
        self.assertIsNotNone(result)
        self.assertTrue("Entity" in result.columns)
        self.assertTrue("Score" in result.columns)

    def test_document_embedding_retrieval(self):
        result = document_embedding_retrieval(self.df, self.query, self.embeddings)
        self.assertIsNotNone(result)
        self.assertTrue("OFFER" in result.columns)
        self.assertTrue("ROUGE_Score" in result.columns)
        self.assertTrue("Jaccard_Score" in result.columns)


if __name__ == "__main__":
    unittest.main()

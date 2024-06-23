# src/analysis/text_analyzer.py

import logging
from typing import List, Dict, Any
import nltk
import spacy
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer

class TextAnalyzer:
    def __init__(self):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        # en_core_web_sm==>This model is a small English model trained on web text
        self.nlp = spacy.load('en_core_web_sm')
        nltk.download('stopwords')

    def analyze_text(self, content: str) -> Dict[str, Any]:
        """
        Performs comprehensive text analysis on the given content.
        
        :param content: The text content to analyze
        :return: A dictionary with text analysis results
        """
        sentences = sent_tokenize(content)
        words = word_tokenize(content)
        clean_words = [word.lower() for word in words if word.isalnum() and word.lower() not in stopwords.words('english')]
        pos_tags = nltk.pos_tag(words)
        named_entities = self._extract_named_entities(content)
        sentiment_score = self._analyze_sentiment(content)
        keywords = self._extract_keywords(clean_words)
        summary = self._generate_summary(content)
        syntax_tree = self._parse_syntax(content)

        analysis_results = {
            'sentences': sentences,
            'words': words,
            'clean_words': clean_words,
            'pos_tags': pos_tags,
            'named_entities': named_entities,
            'sentiment_score': sentiment_score,
            'keywords': keywords,
            'summary': summary,
            'syntax_tree': syntax_tree
        }
        
        return analysis_results

    def _analyze_sentiment(self, content: str) -> Dict[str, float]:
        """
        Analyzes sentiment of the text using VADER sentiment analysis.
        
        :param content: The text content
        :return: A dictionary with sentiment analysis results
        """
        sentiment_scores = self.sentiment_analyzer.polarity_scores(content)
        return sentiment_scores

    def _extract_keywords(self, words: List[str], top_n: int = 5) -> List[str]:
        """
        Extracts keywords from the text based on word frequency.
        
        :param words: List of cleaned words
        :param top_n: Number of top keywords to extract (default: 5)
        :return: List of top keywords
        """
        word_freq = {}
        for word in words:
            if word in word_freq:
                word_freq[word] += 1
            else:
                word_freq[word] = 1
        
        sorted_freq = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        top_keywords = [word for word, freq in sorted_freq[:top_n]]
        return top_keywords

    def _extract_named_entities(self, content: str) -> List[Dict[str, str]]:
        """
        Extracts named entities from the text using SpaCy's named entity recognition (NER).
        
        :param content: The text content
        :return: List of dictionaries with named entity details
        """
        doc = self.nlp(content)
        entities = []
        for ent in doc.ents:
            entities.append({
                'text': ent.text,
                'start_char': ent.start_char,
                'end_char': ent.end_char,
                'label': ent.label_
            })
        return entities

    def _generate_summary(self, content: str, num_sentences: int = 3) -> str:
        """
        Generates a summary of the text using Latent Semantic Analysis (LSA).
        
        :param content: The text content to summarize
        :param num_sentences: Number of sentences in the summary (default: 3)
        :return: A summary of the text
        """
        parser = PlaintextParser.from_string(content, Tokenizer('english'))
        summarizer = LsaSummarizer()
        summary = summarizer(parser.document, num_sentences)
        return ' '.join([str(sentence) for sentence in summary])

    def _parse_syntax(self, content: str) -> Dict[str, Any]:
        """
        Parses the syntax of the text using SpaCy's syntactic parsing.
        
        :param content: The text content
        :return: Dictionary with syntactic parsing details
        """
        doc = self.nlp(content)
        syntax_details = {
            'tokens': [],
            'noun_chunks': [],
            'verbs': []
        }
        
        for token in doc:
            syntax_details['tokens'].append({
                'text': token.text,
                'lemma': token.lemma_,
                'pos': token.pos_,
                'tag': token.tag_,
                'dep': token.dep_
            })
        
        for chunk in doc.noun_chunks:
            syntax_details['noun_chunks'].append({
                'text': chunk.text,
                'root': chunk.root.text,
                'root_dep': chunk.root.dep_
            })
        
        for token in doc:
            if token.pos_ == 'VERB':
                syntax_details['verbs'].append(token.text)
        
        return syntax_details

    def _log_results(self, results: Dict[str, Any]):
        """
        Logs the text analysis results.
        
        :param results: Dictionary with text analysis results
        """
        logging.info(f"Text Analysis Results: {results}")

# Example usage
if __name__ == "__main__":
    analyzer = TextAnalyzer()
    sample_text = """
    Natural Language Processing (NLP) is a subfield of artificial intelligence 
    and linguistics concerned with interactions between computers and human languages.
    """
    text_analysis = analyzer.analyze_text(sample_text)
    analyzer._log_results(text_analysis)

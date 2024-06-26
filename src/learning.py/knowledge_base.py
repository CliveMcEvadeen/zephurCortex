"""
knowledge_base.py

This module is designed to manage the knowledge base for the ZephyrCortex project. It includes functionalities for
storing, retrieving, and updating knowledge using a vector database, ensuring efficient and scalable access to
information. The knowledge base is critical for enabling the learning module to build upon past data and continuously improve.

Features:
- Knowledge Storage: Store textual data as vectors in a vector database for efficient retrieval.
- Knowledge Retrieval: Retrieve relevant knowledge based on similarity searches.
- Knowledge Updating: Update existing knowledge with new information.
- Index Management: Create and manage indices to optimize query performance.
- Data Normalization: Normalize data before storing to ensure consistency.
- Backup and Restore: Backup and restore the knowledge base to prevent data loss.
- Search Optimization: Optimize search queries for faster and more accurate results.
- Knowledge Deletion: Delete specific knowledge entries.
- Knowledge Summarization: Summarize stored knowledge for quick insights.
- Batch Processing: Efficiently process and store large batches of knowledge entries.
- Duplicate Detection: Detect and handle duplicate knowledge entries.
- Knowledge Categorization: Categorize knowledge entries for better organization.
- Metadata Management: Store and manage metadata associated with knowledge entries.

Dependencies:
- numpy
- sklearn
- faiss
- pandas
- SQLAlchemy
- tqdm
- nltk

Example:
    from knowledge_base import KnowledgeBase

    kb = KnowledgeBase()
    kb.store_knowledge("This is some knowledge.")
    results = kb.retrieve_knowledge("Retrieve similar knowledge.")
    kb.update_knowledge("This is updated knowledge.")
"""

import numpy as np
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer
from sqlalchemy import create_engine, Column, String, LargeBinary, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import pandas as pd
import os
from tqdm import tqdm
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('stopwords')
nltk.download('punkt')

Base = declarative_base()

class KnowledgeEntry(Base):
    __tablename__ = 'knowledge_entries'
    id = Column(Integer, primary_key=True)
    text = Column(String)
    vector = Column(LargeBinary)
    metadata = Column(String)

class KnowledgeBase:
    def __init__(self, db_path='sqlite:///knowledge_base.db'):
        self.engine = create_engine(db_path)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.index = None
        self._initialize_index()

    def _initialize_index(self):
        session = self.Session()
        entries = session.query(KnowledgeEntry).all()
        texts = [entry.text for entry in entries]
        if texts:
            self._fit_vectorizer(texts)
            vectors = self.vectorizer.transform(texts).toarray().astype(np.float32)
            self.index = faiss.IndexFlatL2(vectors.shape[1])
            self.index.add(vectors)
        session.close()

    def _fit_vectorizer(self, texts):
        self.vectorizer.fit(texts)

    def store_knowledge(self, text, metadata=None):
        session = self.Session()
        vector = self.vectorizer.transform([text]).toarray().astype(np.float32)
        entry = KnowledgeEntry(text=text, vector=vector.tobytes(), metadata=metadata)
        session.add(entry)
        session.commit()
        if self.index is None:
            self.index = faiss.IndexFlatL2(vector.shape[1])
        self.index.add(vector)
        session.close()

    def retrieve_knowledge(self, query, top_k=5):
        vector = self.vectorizer.transform([query]).toarray().astype(np.float32)
        distances, indices = self.index.search(vector, top_k)
        session = self.Session()
        results = []
        for idx in indices[0]:
            entry = session.query(KnowledgeEntry).get(idx + 1)
            results.append(entry.text)
        session.close()
        return results

    def update_knowledge(self, old_text, new_text, new_metadata=None):
        session = self.Session()
        entry = session.query(KnowledgeEntry).filter_by(text=old_text).first()
        if entry:
            vector = self.vectorizer.transform([new_text]).toarray().astype(np.float32)
            entry.text = new_text
            entry.vector = vector.tobytes()
            if new_metadata:
                entry.metadata = new_metadata
            session.commit()
        session.close()
        self._rebuild_index()

    def delete_knowledge(self, text):
        session = self.Session()
        entry = session.query(KnowledgeEntry).filter_by(text=text).first()
        if entry:
            session.delete(entry)
            session.commit()
        session.close()
        self._rebuild_index()

    def _rebuild_index(self):
        session = self.Session()
        entries = session.query(KnowledgeEntry).all()
        texts = [entry.text for entry in entries]
        vectors = self.vectorizer.transform(texts).toarray().astype(np.float32)
        self.index = faiss.IndexFlatL2(vectors.shape[1])
        self.index.add(vectors)
        session.close()

    def backup_knowledge_base(self, backup_path):
        if os.path.exists(backup_path):
            os.remove(backup_path)
        with self.engine.begin() as conn:
            with open(backup_path, 'wb') as f:
                for chunk in conn.connection.iterdump():
                    f.write(('%s\n' % chunk).encode('utf-8'))

    def restore_knowledge_base(self, backup_path):
        with open(backup_path, 'rb') as f:
            script = f.read().decode('utf-8')
        with self.engine.begin() as conn:
            conn.execute("DROP TABLE IF EXISTS knowledge_entries")
            for statement in script.split(';'):
                if statement.strip():
                    conn.execute(statement)
        self._initialize_index()

    def normalize_data(self, text):
        stop_words = set(stopwords.words('english'))
        word_tokens = word_tokenize(text.lower())
        return ' '.join([word for word in word_tokens if word.isalnum() and word not in stop_words])

    def add_bulk_knowledge(self, texts):
        session = self.Session()
        normalized_texts = [self.normalize_data(text) for text in texts]
        vectors = self.vectorizer.fit_transform(normalized_texts).toarray().astype(np.float32)
        entries = [KnowledgeEntry(text=text, vector=vector.tobytes()) for text, vector in zip(normalized_texts, vectors)]
        session.bulk_save_objects(entries)
        session.commit()
        if self.index is None:
            self.index = faiss.IndexFlatL2(vectors.shape[1])
        self.index.add(vectors)
        session.close()

    def get_all_knowledge(self):
        session = self.Session()
        entries = session.query(KnowledgeEntry).all()
        session.close()
        return [(entry.text, entry.vector) for entry in entries]

    def detect_duplicates(self):
        session = self.Session()
        entries = session.query(KnowledgeEntry).all()
        texts = [entry.text for entry in entries]
        duplicates = set([text for text in texts if texts.count(text) > 1])
        session.close()
        return duplicates

    def categorize_knowledge(self, categories):
        session = self.Session()
        categorized_entries = {category: [] for category in categories}
        entries = session.query(KnowledgeEntry).all()
        for entry in entries:
            for category, keywords in categories.items():
                if any(keyword in entry.text for keyword in keywords):
                    categorized_entries[category].append(entry.text)
                    break
        session.close()
        return categorized_entries

    def summarize_knowledge(self, top_n=5):
        session = self.Session()
        entries = session.query(KnowledgeEntry).all()
        texts = [entry.text for entry in entries]
        vectors = self.vectorizer.transform(texts).toarray().astype(np.float32)
        centroid = np.mean(vectors, axis=0)
        distances = np.linalg.norm(vectors - centroid, axis=1)
        top_indices = distances.argsort()[:top_n]
        summaries = [texts[idx] for idx in top_indices]
        session.close()
        return summaries

# Usage Example
if __name__ == "__main__":
    kb = KnowledgeBase()
    kb.store_knowledge("Artificial Intelligence is the future of technology.", metadata="AI")
    kb.store_knowledge("Machine Learning is a subset of AI.", metadata="ML")
    results = kb.retrieve_knowledge("Tell me about AI")
    print("Retrieved Knowledge:", results)
    kb.update_knowledge("Artificial Intelligence is the future of technology.", "AI is rapidly evolving.")
    duplicates = kb.detect_duplicates()
    print("Detected Duplicates:", duplicates)
    categories = {"Technology": ["technology", "AI", "ML"], "Science": ["physics", "chemistry", "biology"]}
    categorized = kb.categorize_knowledge(categories)
    print("Categorized Knowledge:", categorized)
    summaries = kb.summarize_knowledge()
    print("Knowledge Summaries:", summaries)

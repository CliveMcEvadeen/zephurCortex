"""
data_processing.py

This module provides a comprehensive data processing pipeline for the ZephyrCortex project. 
It includes functions for data cleaning, normalization, feature extraction, dimensionality 
reduction, and more. The module is designed to be flexible and extensible to accommodate 
various data processing needs.

Features:
- Data Cleaning: Remove duplicates, handle missing values, handle outliers.
- Data Normalization: Standard and Min-Max normalization.
- Feature Extraction from Text: TF-IDF vectorization, Bag of Words.
- Dimensionality Reduction: PCA, t-SNE.
- Encoding: One-Hot Encoding for categorical variables.
- Data Augmentation: Generate synthetic data for imbalanced datasets.
- Custom Transformations: Apply user-defined transformations.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from imblearn.over_sampling import SMOTE
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataProcessor:
    def __init__(self):
        self.scaler = None
        self.vectorizer = None
        self.pca = None
        self.tsne = None
        self.encoder = None
        self.data_filter_container=[]

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform basic data cleaning such as removing duplicates, handling missing values, and outliers.
        
        :param df: Input dataframe
        :return: Cleaned dataframe
        """
        logging.info("Cleaning data...")
        df = df.drop_duplicates()
        df = df.dropna()  # Handling missing values can be more sophisticated
        df = df[(np.abs(df - df.mean()) <= (3*df.std())).all(axis=1)]  # Removing outliers
        return df

    def normalize_data(self, df: pd.DataFrame, columns: list, method: str = 'standard') -> pd.DataFrame:
        """
        Normalize numerical columns using specified method.
        
        :param df: Input dataframe
        :param columns: List of columns to normalize
        :param method: Normalization method ('standard' or 'minmax')
        :return: Dataframe with normalized columns
        """
        logging.info("Normalizing data...")
        if method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError("Invalid normalization method")

        df[columns] = self.scaler.fit_transform(df[columns])
        return df

    def encode_categorical_data(self, df: pd.DataFrame, columns: list) -> pd.DataFrame:
        """
        Apply One-Hot Encoding to categorical columns.
        
        :param df: Input dataframe
        :param columns: List of columns to encode
        :return: Dataframe with encoded columns
        """
        logging.info("Encoding categorical data...")
        self.encoder = OneHotEncoder(sparse=False)
        encoded_cols = self.encoder.fit_transform(df[columns])
        encoded_df = pd.DataFrame(encoded_cols, columns=self.encoder.get_feature_names_out(columns))
        return pd.concat([df.drop(columns, axis=1).reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)

    def extract_features_from_text(self, df: pd.DataFrame, column: str, method: str = 'tfidf') -> pd.DataFrame:
        """
        Extract features from text using specified method (TF-IDF or Bag of Words).
        
        :param df: Input dataframe
        :param column: Column containing text data
        :param method: Feature extraction method ('tfidf' or 'bow')
        :return: Dataframe with extracted features
        """
        logging.info("Extracting features from text...")
        if method == 'tfidf':
            self.vectorizer = TfidfVectorizer()
        elif method == 'bow':
            self.vectorizer = CountVectorizer()
        else:
            raise ValueError("Invalid feature extraction method")

        text_features = self.vectorizer.fit_transform(df[column])
        feature_names = self.vectorizer.get_feature_names_out()
        text_df = pd.DataFrame(text_features.toarray(), columns=feature_names)
        return pd.concat([df.reset_index(drop=True), text_df.reset_index(drop=True)], axis=1).drop(columns=[column])

    def apply_pca(self, df: pd.DataFrame, n_components: int) -> pd.DataFrame:
        """
        Apply Principal Component Analysis (PCA) to reduce dimensionality.
        
        :param df: Input dataframe
        :param n_components: Number of principal components
        :return: Dataframe with reduced dimensions
        """
        logging.info("Applying PCA...")
        self.pca = PCA(n_components=n_components)
        principal_components = self.pca.fit_transform(df)
        pca_df = pd.DataFrame(principal_components, columns=[f'PC{i+1}' for i in range(n_components)])
        return pd.concat([df.reset_index(drop=True), pca_df.reset_index(drop=True)], axis=1)

    def apply_tsne(self, df: pd.DataFrame, n_components: int, perplexity: int = 30) -> pd.DataFrame:
        """
        Apply t-Distributed Stochastic Neighbor Embedding (t-SNE) for dimensionality reduction.
        
        :param df: Input dataframe
        :param n_components: Number of components (usually 2 or 3)
        :param perplexity: Perplexity parameter for t-SNE
        :return: Dataframe with t-SNE components
        """
        logging.info("Applying t-SNE...")
        self.tsne = TSNE(n_components=n_components, perplexity=perplexity)
        tsne_components = self.tsne.fit_transform(df)
        tsne_df = pd.DataFrame(tsne_components, columns=[f't-SNE{i+1}' for i in range(n_components)])
        return pd.concat([df.reset_index(drop=True), tsne_df.reset_index(drop=True)], axis=1)

    def augment_data(self, df: pd.DataFrame, target_column: str) -> pd.DataFrame:
        """
        Perform data augmentation using SMOTE for imbalanced datasets.
        
        :param df: Input dataframe
        :param target_column: The target column for augmentation
        :return: Dataframe with augmented data
        """
        logging.info("Augmenting data using SMOTE...")
        smote = SMOTE()
        X = df.drop(columns=[target_column])
        y = df[target_column]
        X_res, y_res = smote.fit_resample(X, y)
        return pd.concat([X_res, y_res], axis=1)

    def apply_custom_transformations(self, df: pd.DataFrame, transformations: list) -> pd.DataFrame:
        """
        Apply custom transformations to the dataframe.
        
        :param df: Input dataframe
        :param transformations: List of custom transformation functions
        :return: Transformed dataframe
        """
        logging.info("Applying custom transformations...")
        for transform in transformations:
            df = transform(df)
        return df

    def process_data(self, df: pd.DataFrame, text_column: str = None, normalize_columns: list = None, 
                     categorical_columns: list = None, n_components: int = None, tsne_components: int = None,
                     target_column: str = None, custom_transformations: list = None) -> pd.DataFrame:
        """
        Comprehensive data processing pipeline.
        
        :param df: Input dataframe
        :param text_column: Column containing text data (if any)
        :param normalize_columns: List of columns to normalize (if any)
        :param categorical_columns: List of categorical columns to encode (if any)
        :param n_components: Number of principal components (if any)
        :param tsne_components: Number of t-SNE components (if any)
        :param target_column: The target column for data augmentation (if any)
        :param custom_transformations: List of custom transformation functions (if any)
        :return: Processed dataframe
        """
        df = self.clean_data(df)
        
        if normalize_columns:
            df = self.normalize_data(df, normalize_columns)
        
        if categorical_columns:
            df = self.encode_categorical_data(df, categorical_columns)
        
        if text_column:
            df = self.extract_features_from_text(df, text_column)
        
        if n_components:
            df = self.apply_pca(df, n_components)
        
        if tsne_components:
            df = self.apply_tsne(df, tsne_components)
        
        if target_column:
            df = self.augment_data(df, target_column)
        
        if custom_transformations:
            df = self.apply_custom_transformations(df, custom_transformations)
        
        return df

# Example usage
if __name__ == "__main__":
    data = {
        'numeric1': [1, 2, 3, 4, 5],
        'numeric2': [5, 4, 3, 2, 1],
        'category': ['A', 'B', 'A', 'B', 'C'],
        'text': ['This is a sentence.', 'Another sentence.', 'Text data here.', 'More text data.', 'Final sentence.']
    }
    df = pd.DataFrame(data)
    
    processor = DataProcessor()
    processed_df = processor.process_data(
        df, 
        text_column='text', 
        normalize_columns=['numeric1', 'numeric2'], 
        categorical_columns=['category'], 
        n_components=2,
        tsne_components=2
    )
    
    print(processed_df)

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler, normalize

from similarity.config import settings


class FeatureBuilder:
    """
    Builds a fixed-size dense feature vector per product.

    Pipeline:
      1. Concatenate all text fields into one string per product
      2. TF-IDF vectorize (sparse, up to 5000 vocab)
      3. TruncatedSVD to 50 dense dims (memory-efficient PCA for sparse input)
      4. L2-normalize the text vectors
      5. Append 2 numeric features: log(price) and rating, MinMax scaled
      Final vector: float32 of shape (n_products, SVD_COMPONENTS + 2)
    """

    def __init__(self):
        self.tfidf = TfidfVectorizer(
            max_features=5000,
            stop_words="english",
            min_df=2,          # drop words appearing in only one product (noise/SKU codes)
            sublinear_tf=True  # log(1+count) instead of raw count — handles keyword stuffing
        )
        self.svd = TruncatedSVD(n_components=settings.SVD_COMPONENTS, random_state=42)
        self.scaler = MinMaxScaler()
        self._price_median = None
        self._rating_median = None

    def build(self, df: pd.DataFrame) -> np.ndarray:
        """Fit all transformers on df and return the full feature matrix."""
        text_features = self._build_text_features(df, fit=True)
        numeric_features = self._build_numeric_features(df, fit=True)
        feature_matrix = np.hstack([text_features, numeric_features])
        return feature_matrix.astype(np.float32)

    def _build_text_features(self, df: pd.DataFrame, fit: bool) -> np.ndarray:
        text_corpus = (
            df["product_name"] + " " +
            df["brand"] + " " +
            df["colour"] + " " +
            df["other_items_customers_buy"]
        ).tolist()

        if fit:
            tfidf_matrix = self.tfidf.fit_transform(text_corpus)
            text_dense = self.svd.fit_transform(tfidf_matrix)
        else:
            tfidf_matrix = self.tfidf.transform(text_corpus)
            text_dense = self.svd.transform(tfidf_matrix)

        # L2 normalize so cosine similarity == dot product in HNSW index
        return normalize(text_dense, norm="l2")

    def _build_numeric_features(self, df: pd.DataFrame, fit: bool) -> np.ndarray:
        if fit:
            self._price_median = df["sales_price"].median()
            self._rating_median = df["rating"].median()

        price_values = df["sales_price"].fillna(self._price_median)
        rating_values = df["rating"].fillna(self._rating_median)

        # log1p handles right-skewed price distribution (most ~₹300, some ~₹7000)
        price_log = np.log1p(price_values.values).reshape(-1, 1)
        rating_col = rating_values.values.reshape(-1, 1)

        numeric_matrix = np.hstack([price_log, rating_col]).astype(np.float32)

        if fit:
            return self.scaler.fit_transform(numeric_matrix).astype(np.float32)
        return self.scaler.transform(numeric_matrix).astype(np.float32)

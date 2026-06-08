import pickle
import re

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler, normalize

from similarity.config import settings

# Number of categorical feature dimensions appended after the numeric block.
# brand (1 dim, label-encoded top-N) + colour (1 dim, hashed bucket).
_CATEGORICAL_DIMS = 2


class FeatureBuilder:
    """
    Builds a fixed-size dense feature vector per product.

    Pipeline:
      1. Concatenate text fields + category tokens into one string per product
      2. TF-IDF vectorize (sparse, up to 5000 vocab)
      3. TruncatedSVD to SVD_COMPONENTS dense dims (memory-efficient PCA for sparse input)
      4. L2-normalize the text vectors
      5. Append 4 numeric features: log(price), rating, log(bestsellers_rank),
         log(weight) — MinMax scaled then multiplied by NUMERIC_WEIGHT
      6. Append 2 categorical features: brand (label-encoded top-N) and colour (hashed)
         — both scaled to [0,1] and down-weighted at NUMERIC_WEIGHT
      Final vector: float32 of shape (n_products, SVD_COMPONENTS + 4 + 2)

    Weight is only 21% populated in this dataset (999999999 sentinel for unknown).
    Missing values are imputed with the median. The spec lists weight as a required
    attribute, so it is included despite the low coverage.

    Category token injection: the child_category label is split from CamelCase into
    words and appended to the text corpus 5 times. This raised same-category hit rate
    from 80% to 96% on this dataset (measured; see docs/tuning.md).
    """

    _CAMEL_RE = re.compile(r'([A-Z])')
    # Top-N brands get a unique integer ID; all others map to 0 (unknown).
    _TOP_BRANDS = 200
    # Number of hash buckets for colour encoding.
    _COLOUR_BUCKETS = 64

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
        self._bsr_median = None
        self._weight_median = None
        # Fitted brand → int mapping (top-N brands; unknown brand → 0)
        self._brand_map: dict = {}

    def build(self, df: pd.DataFrame) -> np.ndarray:
        """Fit all transformers on df and return the full feature matrix."""
        text_features = self._build_text_features(df, fit=True)
        numeric_features = self._build_numeric_features(df, fit=True)
        categorical_features = self._build_categorical_features(df, fit=True)
        feature_matrix = np.hstack([text_features, numeric_features, categorical_features])
        return feature_matrix.astype(np.float32)

    def _build_text_features(self, df: pd.DataFrame, fit: bool) -> np.ndarray:
        # Inject category label as repeated tokens so TF-IDF captures category structure.
        # "WomensKurtasKurtis" → "womens kurtaskurtis " × 5
        # Products without a category label get empty string — no signal lost.
        cat_tokens = df["child_category"].apply(self._category_to_tokens)

        text_corpus = (
            df["product_name"] + " " +
            df["brand"] + " " +
            df["colour"] + " " +
            df["other_items_customers_buy"] + " " +
            cat_tokens
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
            self._price_median  = df["sales_price"].median()
            self._rating_median = df["rating"].median()
            self._bsr_median    = df["bestsellers_rank"].median()
            self._weight_median = df["weight"].median()

        price_log  = np.log1p(df["sales_price"].fillna(self._price_median).values).reshape(-1, 1)
        rating_col = df["rating"].fillna(self._rating_median).values.reshape(-1, 1)
        # log-transform rank: range is 6–2.8M, log maps it to 1–14
        bsr_log    = np.log1p(df["bestsellers_rank"].fillna(self._bsr_median).values).reshape(-1, 1)
        # weight: 21% populated — median-imputed for the rest per spec requirement
        weight_log = np.log1p(df["weight"].fillna(self._weight_median).values).reshape(-1, 1)

        numeric_matrix = np.hstack([price_log, rating_col, bsr_log, weight_log]).astype(np.float32)

        scaled = (
            self.scaler.fit_transform(numeric_matrix) if fit
            else self.scaler.transform(numeric_matrix)
        )
        # Down-weight numeric dims so price/rating inform but don't dominate text.
        # Measured: w=0.3 gives 80% same-category hit rate vs 75% at w=1.0.
        return (scaled * settings.NUMERIC_WEIGHT).astype(np.float32)

    def _build_categorical_features(self, df: pd.DataFrame, fit: bool) -> np.ndarray:
        """
        Encode brand and colour as dedicated scalar dimensions.

        Brand: label-encode the top-_TOP_BRANDS brands; all others map to 0.
          Scaled to [0, 1] so the integer IDs don't dominate by magnitude.
        Colour: hash the normalised colour string into _COLOUR_BUCKETS buckets,
          then scale to [0, 1]. Hashing avoids a large one-hot matrix while
          still giving each colour a stable numeric identity independent of
          which colours appear in the text corpus.

        Both dims are down-weighted by NUMERIC_WEIGHT to match the numeric block.
        """
        if fit:
            top_brands = (
                df["brand"]
                .str.lower().str.strip()
                .value_counts()
                .head(self._TOP_BRANDS)
                .index.tolist()
            )
            # 1-indexed so unknown brand → 0 is distinguishable from the first brand
            self._brand_map = {b: i + 1 for i, b in enumerate(top_brands)}

        brand_ids = (
            df["brand"].str.lower().str.strip()
            .map(lambda b: self._brand_map.get(b, 0))
            .values.reshape(-1, 1)
            .astype(np.float32)
        )
        # Normalise to [0, 1]: max is _TOP_BRANDS (the highest assigned ID)
        brand_scaled = brand_ids / max(len(self._brand_map), 1)

        colour_hashed = (
            df["colour"].str.lower().str.strip()
            .map(lambda c: hash(c) % self._COLOUR_BUCKETS if c else 0)
            .values.reshape(-1, 1)
            .astype(np.float32)
        )
        colour_scaled = colour_hashed / self._COLOUR_BUCKETS

        categorical = np.hstack([brand_scaled, colour_scaled]).astype(np.float32)
        return (categorical * settings.NUMERIC_WEIGHT).astype(np.float32)

    @classmethod
    def _category_to_tokens(cls, category: object) -> str:
        """
        Convert "WomensKurtasKurtis" → "womens kurtas kurtis " repeated 5 times.
        Repeated injection boosts TF-IDF weight for the category signal.
        """
        if not category or not isinstance(category, str):
            return ""
        words = cls._CAMEL_RE.sub(r' \1', category).strip().lower()
        return (words + " ") * 5

    def save(self, path: str) -> None:
        """Pickle the fitted transformers (TF-IDF, SVD, scaler, medians, brand map)."""
        state = {
            "tfidf": self.tfidf,
            "svd": self.svd,
            "scaler": self.scaler,
            "price_median": self._price_median,
            "rating_median": self._rating_median,
            "bsr_median": self._bsr_median,
            "weight_median": self._weight_median,
            "brand_map": self._brand_map,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, path: str) -> "FeatureBuilder":
        """Restore a previously fitted FeatureBuilder from disk."""
        with open(path, "rb") as f:
            state = pickle.load(f)
        obj = cls.__new__(cls)
        obj.tfidf = state["tfidf"]
        obj.svd = state["svd"]
        obj.scaler = state["scaler"]
        obj._price_median = state["price_median"]
        obj._rating_median = state["rating_median"]
        obj._bsr_median = state["bsr_median"]
        obj._weight_median = state.get("weight_median")  # graceful for older caches
        obj._brand_map = state.get("brand_map", {})      # graceful for older caches
        return obj

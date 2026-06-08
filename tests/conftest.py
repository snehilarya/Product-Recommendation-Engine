import pytest
import similarity.engine as engine
from similarity.config import settings
from similarity.feature_builder import FeatureBuilder
from similarity.index import SimilarityIndex


@pytest.fixture(scope="session", autouse=True)
def init_engine():
    """Initialize the engine once for the entire test session."""
    engine.initialize(settings.DATA_PATH)


@pytest.fixture(scope="session")
def df(init_engine):
    """Return the cleaned DataFrame held by the engine — avoids a second load."""
    if engine._df is not None:
        return engine._df
    # Cache-restore path doesn't retain the raw df; fall back to reloading.
    from similarity.engine import _ensure_data_file
    from similarity.data_loader import load_products
    _ensure_data_file(settings.DATA_PATH)
    loaded_df, _, _ = load_products(settings.DATA_PATH)
    return loaded_df


@pytest.fixture(scope="session")
def id_to_index(init_engine):
    return engine._id_to_index


@pytest.fixture(scope="session")
def index_to_id(init_engine):
    return engine._index_to_id


@pytest.fixture(scope="session")
def first_product_id(index_to_id):
    return index_to_id[0]


@pytest.fixture(scope="session")
def built_features(df):
    builder = FeatureBuilder()
    feature_matrix = builder.build(df)
    return feature_matrix, builder


@pytest.fixture(scope="session")
def built_index(df, built_features):
    feature_matrix, _ = built_features
    idx = SimilarityIndex(dim=feature_matrix.shape[1], max_elements=len(df))
    idx.build(feature_matrix)
    return idx

import pytest
import similarity.engine as engine
from similarity.data_loader import load_products
from similarity.feature_builder import FeatureBuilder
from similarity.index import SimilarityIndex
from similarity.config import settings # <-- We import our shiny new settings!

@pytest.fixture(scope="session", autouse=True)
def init_engine():
    """Initialize the engine once for the entire test session."""
    # Use the dynamic settings path!
    engine.initialize(settings.DATA_PATH)

@pytest.fixture(scope="session")
def loaded_data():
    # Use the dynamic settings path!
    df, id_to_index, index_to_id = load_products(settings.DATA_PATH)
    return df, id_to_index, index_to_id

@pytest.fixture(scope="session")
def df(loaded_data):
    return loaded_data[0]

@pytest.fixture(scope="session")
def id_to_index(loaded_data):
    return loaded_data[1]

@pytest.fixture(scope="session")
def index_to_id(loaded_data):
    return loaded_data[2]

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
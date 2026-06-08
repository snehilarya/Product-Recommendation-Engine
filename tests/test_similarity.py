import time
import pytest
import numpy as np


class TestDataLoader:

    def test_loads_at_least_1000_products(self, df):
        assert len(df) > 1000

    def test_required_columns_are_present(self, df):
        required = ["uniq_id", "product_name", "brand", "colour",
                    "sales_price", "rating", "weight"]
        for col in required:
            assert col in df.columns, f"Missing column: {col}"

    def test_no_null_unique_ids(self, df):
        assert df["uniq_id"].isna().sum() == 0

    def test_no_duplicate_unique_ids(self, id_to_index, index_to_id):
        assert len(id_to_index) == len(index_to_id)

    def test_id_to_index_maps_correctly(self, id_to_index, index_to_id):
        first_id = index_to_id[0]
        assert id_to_index[first_id] == 0

    def test_brand_and_colour_have_no_nulls(self, df):
        # null-filled to empty string so TF-IDF concat doesn't break
        assert df["brand"].isna().sum() == 0
        assert df["colour"].isna().sum() == 0

    def test_bestsellers_rank_extracted(self, df):
        # 83% of products have a rank in product_details — spot-check it's numeric
        filled = df["bestsellers_rank"].dropna()
        assert len(filled) > 20000
        assert filled.min() >= 1

    def test_child_category_extracted(self, df):
        filled = df["child_category"].dropna()
        assert len(filled) > 20000
        assert filled.iloc[0].isidentifier() or len(filled.iloc[0]) > 3


class TestFeatureBuilder:

    def test_output_shape_is_correct(self, df, built_features):
        from similarity.config import settings
        feature_matrix, builder = built_features
        expected_dims = settings.SVD_COMPONENTS + 3  # text dims + price + rating + bsr
        assert feature_matrix.shape == (len(df), expected_dims)

    def test_output_dtype_is_float32(self, df, built_features):
        feature_matrix, _ = built_features
        assert feature_matrix.dtype == np.float32

    def test_no_nan_or_inf_in_features(self, built_features):
        feature_matrix, _ = built_features
        assert not np.isnan(feature_matrix).any()
        assert not np.isinf(feature_matrix).any()

    def test_numeric_features_are_scaled(self, built_features):
        from similarity.config import settings
        feature_matrix, _ = built_features
        # last 2 cols are MinMaxScaled then multiplied by NUMERIC_WEIGHT
        numeric_part = feature_matrix[:, -2:]
        assert numeric_part.min() >= 0.0
        assert numeric_part.max() <= settings.NUMERIC_WEIGHT + 1e-5


class TestHNSWIndex:

    def test_query_returns_exact_k_results(self):
        from similarity.index import SimilarityIndex
        dim = 10
        n_items = 100
        rng = np.random.default_rng(seed=0)
        fake_vectors = rng.random((n_items, dim)).astype(np.float32)

        idx = SimilarityIndex(dim=dim, max_elements=n_items)
        idx.build(fake_vectors)

        results = idx.query(vector=fake_vectors[0], k=5, exclude_index=0)
        assert len(results) == 5

    def test_query_excludes_the_query_product(self):
        from similarity.index import SimilarityIndex
        dim = 10
        n_items = 100
        rng = np.random.default_rng(seed=1)
        fake_vectors = rng.random((n_items, dim)).astype(np.float32)

        idx = SimilarityIndex(dim=dim, max_elements=n_items)
        idx.build(fake_vectors)

        query_row = 7
        results = idx.query(vector=fake_vectors[query_row], k=5, exclude_index=query_row)
        assert query_row not in results

    def test_query_returns_valid_row_indices(self, built_features, df):
        from similarity.index import SimilarityIndex
        feature_matrix, _ = built_features
        idx = SimilarityIndex(dim=feature_matrix.shape[1], max_elements=len(df))
        idx.build(feature_matrix)

        results = idx.query(vector=feature_matrix[0], k=10, exclude_index=0)
        for row_index in results:
            assert 0 <= row_index < len(df)


class TestEngine:

    def test_find_similar_returns_correct_count(self, first_product_id):
        import similarity.engine as engine
        results = engine.find_similar_products(first_product_id, num_similar=5)
        assert len(results) == 5

    def test_find_similar_excludes_query_product(self, first_product_id):
        import similarity.engine as engine
        results = engine.find_similar_products(first_product_id, num_similar=10)
        assert first_product_id not in results

    def test_find_similar_returns_valid_product_ids(self, first_product_id, id_to_index):
        import similarity.engine as engine
        results = engine.find_similar_products(first_product_id, num_similar=5)
        for product_id in results:
            assert product_id in id_to_index

    def test_unknown_product_id_raises_key_error(self):
        import similarity.engine as engine
        with pytest.raises(KeyError):
            engine.find_similar_products("this-id-does-not-exist", num_similar=5)

    def test_same_query_returns_same_results(self, first_product_id):
        import similarity.engine as engine
        first_call = engine.find_similar_products(first_product_id, num_similar=5)
        second_call = engine.find_similar_products(first_product_id, num_similar=5)
        assert first_call == second_call

    def test_product_count_matches_dataframe(self, df):
        import similarity.engine as engine
        assert engine.product_count() == len(df)

    def test_price_band_filter_removes_cross_segment_results(self, df, id_to_index):
        import similarity.engine as engine

        # Find a product with a known price so we can check the filter works
        priced = df[df["sales_price"].notna()].iloc[0]
        query_id = priced["uniq_id"]
        query_price = priced["sales_price"]

        results = engine.find_similar_products(query_id, num_similar=10)

        for result_id in results:
            result_price = df.loc[id_to_index[result_id], "sales_price"]
            if result_price is None or (isinstance(result_price, float) and np.isnan(result_price)):
                continue  # unknown price — always allowed through
            ratio = result_price / query_price
            assert (1.0 / 3.0) <= ratio <= 3.0, (
                f"Price band violated: query={query_price}, result={result_price}, ratio={ratio:.2f}"
            )


class TestAPI:

    @pytest.fixture(scope="class")
    def client(self):
        from fastapi.testclient import TestClient
        from app import app
        with TestClient(app, raise_server_exceptions=True) as test_client:
            yield test_client

    def test_health_endpoint_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["products_loaded"] > 0

    def test_find_similar_returns_200_and_list(self, client, first_product_id):
        response = client.get(
            "/find_similar_products",
            params={"product_id": first_product_id, "num_similar": 5}
        )
        assert response.status_code == 200
        results = response.json()
        assert isinstance(results, list)
        assert len(results) == 5

    def test_find_similar_result_ids_are_strings(self, client, first_product_id):
        response = client.get(
            "/find_similar_products",
            params={"product_id": first_product_id, "num_similar": 3}
        )
        for product_id in response.json():
            assert isinstance(product_id, str)

    def test_unknown_product_returns_404(self, client):
        response = client.get(
            "/find_similar_products",
            params={"product_id": "does-not-exist-xyz", "num_similar": 5}
        )
        assert response.status_code == 404

    def test_num_similar_zero_returns_422(self, client, first_product_id):
        response = client.get(
            "/find_similar_products",
            params={"product_id": first_product_id, "num_similar": 0}
        )
        assert response.status_code == 422

    def test_busy_server_returns_503(self, first_product_id):
        # Exhaust the semaphore manually then verify the middleware returns 503
        import app as app_module
        # Drain all slots
        acquired = []
        for _ in range(app_module._MAX_CONCURRENT):
            if app_module._semaphore.acquire(blocking=False):
                acquired.append(True)
        try:
            from fastapi.testclient import TestClient
            from app import app
            with TestClient(app, raise_server_exceptions=False) as c:
                response = c.get(
                    "/find_similar_products",
                    params={"product_id": first_product_id, "num_similar": 5}
                )
            assert response.status_code == 503
        finally:
            for _ in acquired:
                app_module._semaphore.release()


class TestPerformance:

    def test_cached_query_is_fast(self, first_product_id):
        import similarity.engine as engine

        # warm the cache
        engine.find_similar_products(first_product_id, num_similar=10)

        start = time.perf_counter()
        engine.find_similar_products(first_product_id, num_similar=10)
        elapsed_ms = (time.perf_counter() - start) * 1000

        # cached result should be well under 1ms
        assert elapsed_ms < 1.0, f"Cached query took {elapsed_ms:.2f}ms — expected < 1ms"

    def test_uncached_query_is_under_50ms(self, index_to_id):
        import similarity.engine as engine

        # use a product id we haven't queried yet
        some_product_id = index_to_id[500]

        start = time.perf_counter()
        engine.find_similar_products(some_product_id, num_similar=10)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < 50.0, f"HNSW query took {elapsed_ms:.2f}ms — expected < 50ms"

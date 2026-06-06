# Product Similarity Search

A FastAPI microservice that returns similar Amazon fashion products given a product ID. Built as part of the SAP CX II technical exercise.

This service utilizes an **early-fusion monolithic architecture** with dynamic **business-logic guardrails** to achieve sub-millisecond similarity search while strictly respecting user price intent.

## How it works

At startup, the service loads ~30k Amazon fashion products, builds a vector index in memory, and caches metadata:

1. **Text features** — Product name, brand, colour, and description are concatenated into one string per product and vectorized with TF-IDF (5000-token vocabulary, `min_df=2` to drop single-occurrence noise, `sublinear_tf=True` to dampen keyword-stuffed descriptions).
2. **Dimensionality reduction** — `TruncatedSVD` reduces the sparse TF-IDF matrix to 50 dense dimensions (equivalent to PCA, but works directly on sparse input to avoid densifying a 600 MB matrix).
3. **Numeric features** — `sales_price` (log-transformed, then Min-Max scaled) and `rating` (Min-Max scaled) are appended as 2 more dimensions → 52-dim float32 vector per product.
4. **HNSW index** — All 52-dim vectors are inserted into an `hnswlib` cosine-space index (`M=16`, `ef_construction=200`).
5. **Price Guardrails (Post-Filtering)** — To prevent text descriptions from overpowering the price (e.g., matching a $15 watch to a $500 query), the engine fetches an expanded candidate pool (`num_similar * 10`) from the graph and applies a strict Python-level filter to ensure recommendations fall within a 50%–200% price tolerance of the target item.
6. **LRU cache** — Query results are cached by `(product_id, num_similar)` — repeated identical requests return instantly.

## Configuration (.env)

The service is fully configurable via environment variables using `pydantic-settings`. Create a `.env` file in the root directory:

DATA_PATH=data/marketing_sample_for_amazon_com-amazon_fashion_products.ldjson
SVD_COMPONENTS=50
HNSW_M=16
HNSW_EF_CONSTRUCTION=200
HNSW_EF_QUERY=50
PRICE_TOLERANCE_LOWER=0.5
PRICE_TOLERANCE_UPPER=2.0

## Running locally

1. Create and activate a virtual environment:
   python3 -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`

2. Install dependencies:
   pip install -r requirements.txt

3. Ensure your .env file is created and points to the correct DATA_PATH.

4. Start the server:
   uvicorn app:app --host 0.0.0.0 --port 8000

*The server utilizes structured logging and takes ~15–20 seconds to start while it builds the index.*

## Running with Docker (Recommended)

The easiest way to run the service is via the provided `docker-compose.yml`.

docker compose up --build

This automatically mounts the local `./data` folder and exposes the API on port `8000`.

## API

### `GET /find_similar_products`

Returns a list of similar product IDs dynamically filtered by price tolerance.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `product_id` | string | required | `uniq_id` from the dataset |
| `num_similar` | int | 5 | Number of results (1–200) |

Example:
curl "http://localhost:8000/find_similar_products?product_id=26d41bdc1495de290bc8e6062d927729&num_similar=5"

**404** if `product_id` is not in the dataset. **422** if `num_similar` is out of range.

### `GET /health`

Example:
curl http://localhost:8000/health

### Swagger UI
Interactive API docs are available at `http://localhost:8000/docs` after startup.

## Running tests

Run the test suite:
pytest tests/ -v

The test suite covers: data loading and cleaning, feature pipeline correctness (shape, dtype, no NaN), HNSW index behaviour, dynamic price-filtering logic, engine caching, API endpoints, and latency checks (cached query < 1ms, HNSW query < 50ms).

## Design decisions

| Decision | Chosen | Considered | Why |
|---|---|---|---|
| **Architecture** | Monolithic In-Memory | Two-Stage Decoupled Vector DB | Perfect for <100k items. Achieves sub-millisecond latency without the operational overhead of external databases. (Clear migration path to Qdrant/Milvus at 10M+ scale). |
| **Text-Dominance Mitigation** | Post-Filtering (Hard Filter) | Feature Multipliers / One-Hot Encoding | Strictly safeguards user intent (price) using business logic without distorting the vector space geometry or relying purely on mathematical weights. |
| **Text similarity** | TF-IDF | Sentence-transformers | No model download required, highly memory efficient, and perfectly interpretable for this specific dataset. |
| **Dimensionality reduction** | TruncatedSVD | Regular PCA | Works on sparse matrices directly — avoids 600 MB memory densification crash. |
| **ANN index** | hnswlib | FAISS | Simpler API for in-process application use; FAISS/Milvus would be the choice for billion-scale deployment. |
| **Configuration** | Pydantic Settings | Hardcoded Variables | Ensures the microservice is instantly tunable for DevOps teams via environment variables in Kubernetes/Docker. |
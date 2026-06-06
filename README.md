# Product Similarity Search

A FastAPI microservice that returns similar Amazon fashion products given a product ID. Built as part of the SAP CX II technical exercise.

## How it works

At startup the service loads ~30k Amazon fashion products and builds a vector index:

1. **Text features** — product name, brand, colour, and description are concatenated into one string per product and vectorized with TF-IDF (5000-token vocabulary, `min_df=2` to drop single-occurrence noise, `sublinear_tf=True` to dampen keyword-stuffed descriptions)
2. **Dimensionality reduction** — TruncatedSVD reduces the sparse TF-IDF matrix to 50 dense dimensions (equivalent to PCA, but works on sparse input directly — avoids densifying a 600 MB matrix)
3. **Numeric features** — `sales_price` (log-transformed, then Min-Max scaled) and `rating` (Min-Max scaled) appended as 2 more dimensions → 52-dim float32 vector per product
4. **HNSW index** — all 52-dim vectors are inserted into an hnswlib cosine-space index (`M=16`, `ef_construction=200`) for approximate nearest-neighbour search at sub-millisecond latency
5. **LRU cache** — query results are cached by `(product_id, num_similar)` — repeated identical requests return instantly

## Running locally

```bash
# Install dependencies
pip install -r requirements.txt

# Start the server — the dataset is bundled in data/archive.zip and extracted automatically
uvicorn app:app --host 0.0.0.0 --port 8000
```

The server takes ~15–20 seconds to start while it builds the index. On first startup it also extracts `data/archive.zip` (~12 MB) into the `data/` folder.

## Running with Docker

```bash
docker build -t similarity-search .
docker run -p 8000:8000 similarity-search
```

The dataset is copied into the image at build time — no volume mount or external data needed.

## API

### `GET /find_similar_products`

Returns a list of similar product IDs.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `product_id` | string | required | `uniq_id` from the dataset |
| `num_similar` | int | 5 | Number of results (1–200) |

```bash
curl "http://localhost:8000/find_similar_products?product_id=26d41bdc1495de290bc8e6062d927729&num_similar=5"
# ["abc123...", "def456...", ...]
```

**404** if `product_id` is not in the dataset. **422** if `num_similar` is out of range.

### `GET /health`

```bash
curl http://localhost:8000/health
# {"status": "ok", "products_loaded": 30000}
```

### Swagger UI

Interactive API docs at `http://localhost:8000/docs` after startup.

## Running tests

```bash
pytest tests/ -v
```

All 28 tests cover: data loading and cleaning, feature pipeline correctness (shape, dtype, no NaN), HNSW index behaviour, engine caching, API endpoints, and latency (cached query < 1ms, HNSW query < 50ms).

## Design decisions

| Decision | Chosen | Considered | Why |
|---|---|---|---|
| Text similarity | TF-IDF | Sentence-transformers | No model download, fast, interpretable |
| Dimensionality reduction | TruncatedSVD | Regular PCA | Works on sparse matrices directly — avoids 600 MB densification |
| ANN index | hnswlib | FAISS | Simpler API for in-process use at this scale; FAISS is the better choice at billion-scale |
| Feature combination | Concat + cosine | Separate indexes + score fusion | Single explainable pipeline |
| Caching | `functools.lru_cache` | Redis | No extra services needed for single-pod deployment |
| Index lifetime | Build on startup | Persist to disk | No build step or file management; adds ~15s startup time |

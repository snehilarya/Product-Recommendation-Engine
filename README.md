# Product Similarity Search

A FastAPI microservice that returns similar Amazon fashion products given a product ID. Built as part of the SAP CX II technical exercise.

## How it works

At startup the service loads ~30k Amazon fashion products and builds a vector index in memory. Here's the full pipeline:

1. **Text features** — product name, brand, colour, and "customers also bought" text are concatenated into one string per product. We also inject the product's Amazon category label (e.g. "WomensKurtasKurtis") repeated 5 times — this turned out to be the biggest single accuracy improvement, taking same-category hit rate from 80% to 96%. Everything goes through TF-IDF (5000-token vocabulary).

2. **Dimensionality reduction** — TruncatedSVD squashes the sparse TF-IDF matrix down to 75 dense dimensions. We use TruncatedSVD instead of regular PCA because PCA would first need to convert the sparse matrix to dense (that's ~600 MB for this dataset). TruncatedSVD does the same thing but works on sparse input directly.

3. **Numeric features** — `sales_price` (log-transformed to handle the right skew — most items are ₹300–₹600 but some are ₹8000+), `rating`, and `bestsellers_rank` (also log-transformed, range is 6 to 2.8M) are scaled and appended as 3 more dimensions. These are down-weighted at 0.3× so they inform but don't dominate the text signal → 78-dim float32 vector per product.

4. **HNSW index** — all vectors go into an hnswlib cosine-space index for approximate nearest-neighbour search. Queries run in ~0.02ms.

5. **Price band filter** — after HNSW returns candidates, we throw out anything priced more than 3× higher or lower than the query product. Prevents a ₹15 plastic watch from being recommended next to a ₹500 leather one just because both say "black" and "watch".

6. **Query cache** — results are stored in a bounded in-process cache (capped at 10,000 entries, FIFO eviction). Repeated identical queries return instantly without hitting the HNSW index.

## Running locally

```bash
pip install -r requirements.txt

uvicorn app:app --host 0.0.0.0 --port 8000
```

First startup takes about 2–3 seconds to build the index and saves it to `.index_cache/`. Every restart after that loads from disk and is ready in ~50ms instead. The cache directory is versioned by config hash — if you change `SVD_COMPONENTS` or `NUMERIC_WEIGHT`, the app automatically detects the mismatch and rebuilds.

The dataset is bundled as `data/archive.zip` and extracted automatically on first run — no manual data setup needed.

## Running with Docker

```bash
docker build -t similarity-search .
docker run -p 8000:8000 similarity-search
```

## API

### `GET /find_similar_products`

Returns a list of similar product IDs.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `product_id` | string | required | `uniq_id` from the dataset |
| `num_similar` | int | 5 | How many results to return (1–200) |

```bash
curl "http://localhost:8000/find_similar_products?product_id=26d41bdc1495de290bc8e6062d927729&num_similar=5"
# ["abc123...", "def456...", ...]
```

**404** if the product ID isn't in the dataset. **422** if `num_similar` is out of range. **503** if the server is handling too many requests at once — just retry.

### `GET /health`

```bash
curl http://localhost:8000/health
# {"status": "ok", "products_loaded": 30000}
```

Use this as the Kubernetes liveness/readiness probe. It returns `products_loaded: 0` if the engine hasn't finished initializing yet.

### Swagger UI

Interactive docs at `http://localhost:8000/docs` once the server is running.

## Running tests

```bash
pytest tests/ -v
```

31 tests covering: data loading and cleaning, feature pipeline (shape, dtype, no NaN/inf), HNSW index behaviour, engine caching, API endpoints (200/404/422/503), and latency (cached query < 1ms, HNSW query < 50ms).

## Architecture decisions

### What we picked and why

| Decision | Chosen | Why |
|---|---|---|
| Text similarity | TF-IDF + TruncatedSVD | No model download, fast at startup, explainable |
| ANN index | hnswlib HNSW | Sub-millisecond queries, simple in-process API — FAISS would be better at billion-scale but adds complexity |
| Feature combination | Concat + cosine | Single pipeline, easy to explain |
| Index persistence | Save to `.index_cache/` on first build | Cold start drops from 2,600ms to 51ms on restart |
| Query caching | Bounded in-process dict (10k entries, FIFO) | No extra services needed; fast enough for single-pod deployment |
| Backpressure | Semaphore (max 50 concurrent) → 503 | Prevents the server from queueing requests into memory exhaustion under burst load |

### What we considered and skipped

- **Sentence transformers / BERT embeddings** — would give better semantic understanding but startup time goes from 3 seconds to several minutes and the model is 400MB+. For a 30k-product dataset where product names are already quite descriptive, TF-IDF captures most of the signal.

- **FAISS** — more battle-tested at scale, better GPU support. For 30k products in a single process, hnswlib is simpler and equally fast.

- **Separate image similarity index** — the dataset has image URLs but most are broken (2020 Amazon data). Worth adding if fresh images were available; the architecture already supports extending the feature vector.

- **Shared cache across replicas** — if running multiple Kubernetes replicas, each pod has its own independent cache. A shared Redis cache would unify hot query results across pods, but adds an external dependency and operational overhead. For this scale it's not worth it.

### Accuracy improvements (measured on this dataset)

We tested every feature change against same-category hit rate — for a given product, what percentage of the top-3 results are from the same Amazon category.

| Version | Same-category hit rate |
|---|---|
| Baseline (SVD=50, price+rating) | 75% |
| SVD tuned to 75 | 80% |
| + Category tokens in TF-IDF corpus | 95% |
| + Amazon Bestsellers Rank | 96% |

The category token injection was by far the biggest win. Every product has an Amazon category label (`WomensKurtasKurtis`, `MensT_Shirts`, etc.) that we were ignoring. Splitting it into words and repeating it 5 times in the text corpus gives TF-IDF a very strong "this is what kind of thing this is" signal.

# Product Similarity Search

A FastAPI microservice that returns similar Amazon fashion products given a product ID. Built as part of the SAP CX II technical exercise.

## How it works

At startup the service loads ~26k Amazon fashion products (30k raw records minus ~4k near-duplicate SKUs removed at load time) and builds a vector index in memory. Here's the full pipeline:

1. **Text features** — product name, brand, colour, and "customers also bought" text are concatenated into one string per product. We also inject the product's Amazon category label (e.g. "WomensKurtasKurtis") repeated 5 times — this turned out to be the biggest single accuracy improvement, taking same-category hit rate from 80% to 96%. Everything goes through TF-IDF (5000-token vocabulary).

2. **Dimensionality reduction** — TruncatedSVD squashes the sparse TF-IDF matrix down to 75 dense dimensions. We use TruncatedSVD instead of regular PCA because PCA would first need to convert the sparse matrix to dense (that's ~600 MB for this dataset). TruncatedSVD does the same thing but works on sparse input directly.

3. **Numeric features** — four numeric attributes from the spec are appended as additional dimensions, each log-transformed where skewed and MinMax scaled then down-weighted at 0.3×:
   - `sales_price` — log-transformed (right-skewed: most ₹300–₹600, some ₹8000+)
   - `rating` — direct scale
   - `bestsellers_rank` — log-transformed (range 6 to 2.8M)
   - `weight` — log-transformed; only 21% of products have a real value, the rest are median-imputed

4. **Categorical features** — two additional feature groups appended after the numeric block, both down-weighted at 0.3×:
   - `brand` — top-50 brands by frequency get a unique label-encoded ID; all others map to 0. 50 was chosen after benchmarking: same-category hit rate is flat (within 0.08%) across top-50/100/150/200, so using more brands adds noise rather than signal.
   - `colour` — 12 canonical colour binary dimensions (blue, black, red, white, pink, green, grey, yellow, orange, purple, brown, multi). The raw colour string (e.g. `"black|blue|red"`) is split and each word matched to a canonical group — a product can activate multiple dims simultaneously. 80% of products have no colour value; these get all-zero colour dims. Hash-bucketing was considered and rejected: 90% of the 4,671 unique colour strings appear only once, making hashing indistinguishable from noise.

   → **92-dim float32 vector per product** (75 text + 4 numeric + 1 brand + 12 colour).

5. **HNSW index** — all vectors go into an hnswlib cosine-space index for approximate nearest-neighbour search. Queries run in ~0.02ms.

6. **Price band filter** — after HNSW returns candidates, we throw out anything priced more than 3× higher or lower than the query product. Prevents a ₹15 plastic watch from being recommended next to a ₹500 leather one just because both say "black" and "watch".

7. **Near-duplicate removal** — products with the same name, brand, and price are deduplicated at load time. Without this, the same item listed multiple times under different seller IDs would fill all top-N slots. Empty-brand products are excluded from deduplication (two unbranded items with the same name and price may be genuinely different).

8. **Query cache** — results are stored in a bounded in-process cache (capped at 10,000 entries, FIFO eviction, thread-safe). Repeated identical queries return instantly without hitting the HNSW index.

## Running locally

```bash
pip install -r requirements.txt

uvicorn app:app --host 0.0.0.0 --port 8000
```

First startup takes about 2–3 seconds to build the index and saves it to `.index_cache/`. Every restart after that loads from disk and is ready in ~1 second instead. The cache directory is versioned by config hash — if you change `SVD_COMPONENTS`, `NUMERIC_WEIGHT`, `HNSW_M`, `HNSW_EF_CONSTRUCTION`, or `FEATURE_VERSION`, the app automatically detects the mismatch and rebuilds.

> **Kubernetes note:** `.index_cache/` is written to the container's local filesystem. In K8s, pod recreation (deploys, node rescheduling, OOM kills) wipes the local filesystem — the fast-load path only applies to in-place restarts. To benefit from caching across pod recreations, mount a `PersistentVolumeClaim` at the path set by the `CACHE_DIR` environment variable.

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
# [{"product_id": "abc123...", "similarity_score": 0.9308}, ...]
```

**404** if the product ID isn't in the dataset. **422** if `num_similar` is out of range. **503** if the server is handling too many requests at once — just retry.

### `GET /health`

```bash
curl http://localhost:8000/health
# {"status": "ok", "products_loaded": 25855}
```

Use this as the Kubernetes liveness/readiness probe. It returns `products_loaded: 0` if the engine hasn't finished initializing yet.

### Swagger UI

Interactive docs at `http://localhost:8000/docs` once the server is running.

## Running tests

```bash
pytest tests/ -v
```

33 tests covering: data loading and cleaning (including weight sentinel handling), feature pipeline (shape, dtype, no NaN/inf), HNSW index behaviour, engine caching, API endpoints (200/404/422/503), and latency (cached query < 1ms, HNSW query < 50ms).

## Architecture decisions

### What we picked and why

| Decision | Chosen | Why |
|---|---|---|
| Text similarity | TF-IDF + TruncatedSVD | No model download, fast at startup, explainable |
| ANN index | hnswlib HNSW | Sub-millisecond queries, simple in-process API. Based on Malkov & Yashunin (2018) — [arxiv.org/abs/1603.09320](https://arxiv.org/abs/1603.09320). FAISS would be better at billion-scale but adds complexity for 30k products. |
| Feature combination | Concat + cosine | Single pipeline, easy to explain |
| Brand encoding | Label-encode top-50 brands | Top-50 covers brands with ≥42 products. Benchmarked flat accuracy across top-50/100/150/200 — using more adds noise. |
| Colour encoding | 12 canonical colour binary dims | 90% of 4,671 unique colour strings appear only once — hashing is indistinguishable from noise. Canonical extraction (blue, black, red, ...) gives honest multi-hot signal. |
| Index persistence | Save to `.index_cache/` on first build | Cold start drops from 2,600ms to 51ms on restart |
| Query caching | Bounded in-process dict (10k entries, FIFO, thread-safe) | No extra services needed; fast enough for single-pod deployment |
| Backpressure | Semaphore (max 50 concurrent) → 503 | Prevents the server from queueing requests into memory exhaustion under burst load |

### What we considered and skipped

- **Sentence transformers / BERT embeddings** — would give better semantic understanding but startup time goes from 3 seconds to several minutes and the model is 400MB+. For a 30k-product dataset where product names are already quite descriptive, TF-IDF captures most of the signal.

- **FAISS** — more battle-tested at scale, better GPU support. For 30k products in a single process, hnswlib is simpler and equally fast.

- **One-hot encoding for brand** — correct encoding but creates a sparse matrix. The right approach is to hstack the one-hot matrix with TF-IDF before TruncatedSVD (which handles sparse input). Label encoding was chosen for simplicity; the accuracy difference is negligible at this scale.

- **Separate image similarity index** — the dataset has image URLs but most are broken (2020 Amazon data). Worth adding if fresh images were available; the architecture already supports extending the feature vector.

- **Shared cache across replicas** — if running multiple Kubernetes replicas, each pod has its own independent cache. A shared Redis cache would unify hot query results across pods, but adds an external dependency and operational overhead. For this scale it's not worth it.

### Accuracy improvements (measured on this dataset)

We tested every feature change against same-category hit rate — for a given product, what percentage of the top-5 results are from the same Amazon category (500-product random sample, seed=42).

| Version | Same-category hit rate |
|---|---|
| Baseline (SVD=50, price+rating) | 75% |
| SVD tuned to 75 | 80% |
| + Category tokens in TF-IDF corpus | 95% |
| + Amazon Bestsellers Rank | 96% |
| + Brand (top-50 label-encoded) + Colour (12 canonical dims) | 94% |

The category token injection was by far the biggest win. Every product has an Amazon category label (`WomensKurtasKurtis`, `MensT_Shirts`, etc.) that we were ignoring. Splitting it into words and repeating it 5 times in the text corpus gives TF-IDF a very strong "this is what kind of thing this is" signal.

The brand and colour features do not improve same-category hit rate — colour especially, since 80% of products have no colour value and a blue kurti vs a blue t-shirt share colour but not category. Their value is in within-category ranking: two black Nike t-shirts should rank higher than a yellow Adidas one, which same-category rate does not measure.

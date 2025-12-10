# Milvus 2.6.7 Native Hybrid Search REST Proxy

A production-grade REST API for Milvus 2.6.7 supporting native hybrid search with text and image embeddings.

## Features

- **Native Hybrid Search** - Uses Milvus 2.6's built-in `hybrid_search()` method
- **Multi-modal Search** - Combine text and image search with intelligent weighting
- **Dynamic Collections** - Search any collection via URL parameter
- **Flexible Filtering** - Support for price, bedroom, property type filters
- **CLIP Integration** - Text-to-image search using CLIP model
- **RRF & Weighted Ranking** - Choose between reciprocal rank fusion or custom weights

## Configuration

### Required Environment Variable

The proxy requires `MILVUS_HOST` to be set:

```bash
# Option 1: Environment variable
export MILVUS_HOST=your-milvus-host
python3 rest-proxy-multimodal.py

# Option 2: Inline
MILVUS_HOST=your-milvus-host python3 rest-proxy-multimodal.py
```

### Optional Environment Variables

```bash
MILVUS_PORT=443  # Default: 443
```

### Examples for Different Milvus Instances

```bash
# Local Milvus
MILVUS_HOST=localhost MILVUS_PORT=19530 python3 rest-proxy-multimodal.py

# Milvus Cloud
MILVUS_HOST=in01-xxxxxxxx.aws-us-west-2.vectordb.zilliz.com python3 rest-proxy-multimodal.py

# Railway Deployment
# Set MILVUS_HOST in Railway Variables tab
```

## API Endpoints

### Hybrid Search
```bash
POST /search/{collection_name}/hybrid
Content-Type: application/json

{
    "text": "modern house with pool",
    "limit": 10,
    "filters": {
        "price_max": 500000,
        "bedrooms_min": 3
    }
}
```

### Text-only Search
```bash
POST /search/{collection_name}
Content-Type: application/json

{
    "text": "kitchen renovation",
    "limit": 5
}
```

### Collection Stats
```bash
GET /stats/{collection_name}
```

### List Collections
```bash
GET /collections
```

### Health Check
```bash
GET /health
```

## Railway Deployment

1. **Set Environment Variables** in Railway dashboard:
   ```
   MILVUS_HOST=your-milvus-host
   ```

2. **OR use Custom Start Command**:
   ```
   MILVUS_HOST=your-milvus-host python3 rest-proxy-multimodal.py
   ```

3. **Deploy** - Railway will automatically build and deploy

## How It Works

1. **Intelligent Weight Distribution**:
   - Text queries: 70% text, 30% image
   - Visual queries ("red roof"): 50/50 split
   - Custom weights available via API

2. **Collection Support**:
   - Dynamic collection names via URL
   - Works with any collection that has `text_embedding` and `image_embedding` fields
   - Supports all Milvus 2.6.7 collection types

3. **Filter Conversion**:
   - Converts JSON filters to Milvus expressions
   - Supports range queries (`_max`, `_min` suffixes)
   - Example: `{"price_max": 500000}` → `price <= 500000`

## Dependencies

- Python 3.8+
- Milvus 2.6.7
- Flask
- sentence-transformers
- torch
- transformers
- pymilvus

## Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run locally
MILVUS_HOST=localhost python3 rest-proxy-multimodal.py
```

## Production Considerations

- **Model Loading**: Models load lazily on first request
- **Performance**: Sub-second response times for typical queries
- **Scalability**: Handles millions of vectors with Milvus clustering
- **Security**: Uses secure connections (TLS) by default
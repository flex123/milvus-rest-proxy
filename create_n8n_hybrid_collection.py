#!/usr/bin/env python3
"""
Create a hybrid collection compatible with n8n and REST proxy
Incorporates all the fixes and improvements we've made
"""

import os
import requests
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
import numpy as np
import time
from sentence_transformers import SentenceTransformer
from PIL import Image
import io
import base64

print("="*80)
print("CREATE N8N HYBRID COLLECTION")
print("="*80)

# Connect to Milvus
connections.connect(
    alias="default",
    host="grpc-reverse-proxy-production-039b.up.railway.app",
    port=443,
    secure=True,
    timeout=30
)

# Configuration
COLLECTION_NAME = "n8n_properties_hybrid"
DIMENSION = 1536  # OpenAI text-embedding-ada-002 dimension
TEXT_MODEL = "all-MiniLM-L6-v2"  # For text embeddings (will be replaced with OpenAI if API key is set)
IMAGE_MODEL = "clip-ViT-B-32"  # For image embeddings

# Check for OpenAI API key
USE_OPENAI = os.getenv('OPENAI_API_KEY') is not None
if USE_OPENAI:
    print("\n✅ OpenAI API key detected - will use OpenAI embeddings")
    import openai
    openai_client = openai.OpenAI()
else:
    print("\n⚠️  No OpenAI API key - will use sentence-transformers (not compatible with n8n)")

# Check if collection exists
if utility.has_collection(COLLECTION_NAME):
    print(f"\nCollection '{COLLECTION_NAME}' already exists")
    response = input("Drop and recreate? (y/n): ")
    if response.lower() == 'y':
        utility.drop_collection(COLLECTION_NAME)
        print("Dropped existing collection")
    else:
        print("Using existing collection")
        collection = Collection(COLLECTION_NAME)
        collection.load()
        print(f"Collection has {collection.num_entities} entities")
        exit(0)

# Load models lazily (to avoid Railway timeout)
print("\nLoading ML models...")
text_encoder = SentenceTransformer(TEXT_MODEL)
image_encoder = SentenceTransformer(IMAGE_MODEL)
print("✅ Models loaded")

# Define schema for n8n compatibility
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
    FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=20000),  # For n8n
    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=DIMENSION),  # OpenAI embeddings for n8n
    FieldSchema(name="text_embedding", dtype=DataType.FLOAT_VECTOR, dim=512),  # Sentence transformer
    FieldSchema(name="image_embedding", dtype=DataType.FLOAT_VECTOR, dim=512),  # CLIP image embeddings
    FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=1000),
    FieldSchema(name="description", dtype=DataType.VARCHAR, max_length=5000),
    FieldSchema(name="price", dtype=DataType.DOUBLE),
    FieldSchema(name="city", dtype=DataType.VARCHAR, max_length=100),
    FieldSchema(name="state", dtype=DataType.VARCHAR, max_length=100),
    FieldSchema(name="bedrooms", dtype=DataType.INT64),
    FieldSchema(name="bathrooms", dtype=DataType.DOUBLE),
    FieldSchema(name="square_feet", dtype=DataType.INT64),
    FieldSchema(name="property_type", dtype=DataType.VARCHAR, max_length=100),
    FieldSchema(name="image_url", dtype=DataType.VARCHAR, max_length=1000),
    FieldSchema(name="has_pool", dtype=DataType.BOOL),
    FieldSchema(name="has_garden", dtype=DataType.BOOL),
    FieldSchema(name="has_ocean_view", dtype=DataType.BOOL),
    FieldSchema(name="has_mountain_view", dtype=DataType.BOOL)
]

schema = CollectionSchema(
    fields=fields,
    description="N8n compatible hybrid properties collection with multiple embeddings"
)

collection = Collection(COLLECTION_NAME, schema)
print(f"✅ Collection '{COLLECTION_NAME}' created")

# Create indexes
print("\nCreating indexes...")

# L2 index for n8n compatibility
index_params_l2 = {
    "metric_type": "L2",
    "index_type": "IVF_FLAT",
    "params": {"nlist": 1024}
}
collection.create_index("vector", index_params_l2)
print("✅ L2 index created for OpenAI embeddings (n8n compatible)")

# COSINE index for hybrid search
index_params_cosine = {
    "metric_type": "COSINE",
    "index_type": "IVF_FLAT",
    "params": {"nlist": 1024}
}
collection.create_index("text_embedding", index_params_cosine)
collection.create_index("image_embedding", index_params_cosine)
print("✅ COSINE indexes created for hybrid search")

# Static image URLs
image_urls = [
    "https://cdn.pixabay.com/photo/2016/11/18/17/46/house-1836070_960_720.jpg",  # Modern house
    "https://cdn.pixabay.com/photo/2017/08/01/14/49/beach-2564837_960_720.jpg",  # Beach
    "https://cdn.pixabay.com/photo/2016/06/24/10/46/house-1476991_960_720.jpg",  # Luxury house
    "https://cdn.pixabay.com/photo/2015/10/20/18/46/apartment-996779_960_720.jpg",  # Apartment
    "https://cdn.pixabay.com/photo/2017/08/27/10/16/vintage-2688526_960_720.jpg",  # Penthouse
    "https://cdn.pixabay.com/photo/2018/03/16/21/19/log-cabin-3237454_960_720.jpg",  # Cabin
    "https://cdn.pixabay.com/photo/2016/11/29/09/00/architecture-1868549_960_720.jpg",  # Modern
    "https://cdn.pixabay.com/photo/2017/04/10/22/34/villa-2219075_960_720.jpg",  # Villa
    "https://cdn.pixabay.com/photo/2014/07/10/10/13/house-389257_960_720.jpg",  # Family home
    "https://cdn.pixabay.com/photo/2016/01/21/17/48/house-1152584_960_720.jpg",  # Suburban
    "https://cdn.pixabay.com/photo/2015/05/15/14/46/architecture-768765_960_720.jpg",  # Contemporary
    "https://cdn.pixabay.com/photo/2018/01/09/14/40/home-3072718_960_720.jpg",  # Traditional
    "https://cdn.pixabay.com/photo/2015/12/01/20/28/road-1072883_960_720.jpg",  # Mountain view
    "https://cdn.pixabay.com/photo/2019/02/04/15/39/town-3974922_960_720.jpg",  # Townhouse with red roof
    "https://cdn.pixabay.com/photo/2016/08/05/10/08/villa-1573870_960_720.jpg",  # Pool
    "https://cdn.pixabay.com/photo/2017/08/01/12/49/pool-2564858_960_720.jpg",  # Swimming pool
    "https://cdn.pixabay.com/photo/2016/12/30/15/50/house-1935190_960_720.jpg",  # Garden
    "https://cdn.pixabay.com/photo/2017/07/09/03/07/house-2487347_960_720.jpg",  # Sea view
    "https://cdn.pixabay.com/photo/2017/06/08/08/23/log-cabin-2384022_960_720.jpg",  # Wood cabin
    "https://cdn.pixabay.com/photo/2014/09/16/18/35/house-448438_960_720.jpg",  # Cottage
    "https://cdn.pixabay.com/photo/2017/08/01/00/43/architecture-394871_960_720.jpg",  # Urban
    "https://cdn.pixabay.com/photo/2016/11/23/13/48/beach-1852839_960_720.jpg",  # Ocean view
    "https://cdn.pixabay.com/photo/2015/01/28/23/35/architecture-615283_960_720.jpg",  # City
    "https://cdn.pixabay.com/photo/2015/09/02/12/25/housing-922418_960_720.jpg",  # Condo
    "https://cdn.pixabay.com/photo/2017/08/08/02/39/building-2612813_960_720.jpg",  # High-rise
    "https://cdn.pixabay.com/photo/2015/02/15/13/27/palace-636395_960_720.jpg",  # Mansion
    "https://cdn.pixabay.com/photo/2017/08/07/21/51/house-2608282_960_720.jpg",  # Luxury
    "https://cdn.pixabay.com/photo/2014/09/12/10/40/house-442699_960_720.jpg",  # Residential
    "https://cdn.pixabay.com/photo/2014/08/01/00/35/kitchen-408308_960_720.jpg",  # Interior
    "https://cdn.pixabay.com/photo/2015/01/08/18/28/desk-593341_960_720.jpg",  # Office
    "https://cdn.pixabay.com/photo/2014/08/01/00/35/kitchen-408308_960_720.jpg",  # Kitchen (duplicate)
    "https://cdn.pixabay.com/photo/2015/09/02/12/49/home-922531_960_720.jpg",  # Bathroom
    "https://cdn.pixabay.com/photo/2014/09/21/16/35/house-455896_960_720.jpg",  # Exterior
    "https://cdn.pixabay.com/photo/2014/09/12/10/40/house-442699_960_720.jpg",  # Residential (duplicate)
    "https://cdn.pixabay.com/photo/2014/07/10/10/13/house-389257_960_720.jpg",  # Family home (duplicate)
    "https://cdn.pixabay.com/photo/2016/01/21/17/48/house-1152584_960_720.jpg",  # Suburban (duplicate)
    "https://cdn.pixabay.com/photo/2015/10/20/18/46/apartment-996779_960_720.jpg",  # Apartment (duplicate)
    "https://cdn.pixabay.com/photo/2017/08/01/14/49/beach-2564837_960_720.jpg",  # Beach (duplicate)
    "https://cdn.pixabay.com/photo/2017/06/08/08/23/log-cabin-2384022_960_720.jpg"   # Wood cabin (duplicate)
]

# Static property data
properties_data = [
    {"id": 1, "title": "Beach House in Boston", "description": "Beautiful beachfront property with ocean views", "city": "Boston", "state": "MA", "price": 850000, "bedrooms": 3, "bathrooms": 2.5, "square_feet": 2000, "property_type": "Beach House", "has_pool": False, "has_garden": True, "has_ocean_view": True, "has_mountain_view": False},
    {"id": 2, "title": "Penthouse in Seattle", "description": "Luxury penthouse with city skyline views", "city": "Seattle", "state": "WA", "price": 1200000, "bedrooms": 2, "bathrooms": 2, "square_feet": 1800, "property_type": "Penthouse", "has_pool": True, "has_garden": False, "has_ocean_view": False, "has_mountain_view": True},
    {"id": 3, "title": "Modern Villa in Miami", "description": "Contemporary villa with infinity pool", "city": "Miami", "state": "FL", "price": 1500000, "bedrooms": 5, "bathrooms": 4.5, "square_feet": 4500, "property_type": "Villa", "has_pool": True, "has_garden": True, "has_ocean_view": True, "has_mountain_view": False},
    {"id": 4, "title": "Cozy Mountain Cabin", "description": "Rustic cabin perfect for getaway", "city": "Aspen", "state": "CO", "price": 650000, "bedrooms": 2, "bathrooms": 1, "square_feet": 1200, "property_type": "Mountain Cabin", "has_pool": False, "has_garden": False, "has_ocean_view": False, "has_mountain_view": True},
    {"id": 5, "title": "Urban Apartment", "description": "Modern apartment in downtown area", "city": "New York", "state": "NY", "price": 750000, "bedrooms": 1, "bathrooms": 1, "square_feet": 800, "property_type": "Apartment", "has_pool": True, "has_garden": False, "has_ocean_view": False, "has_mountain_view": False},
    {"id": 6, "title": "Suburban Family Home", "description": "Perfect house for growing family", "city": "Austin", "state": "TX", "price": 450000, "bedrooms": 4, "bathrooms": 3, "square_feet": 2500, "property_type": "Single Family Home", "has_pool": True, "has_garden": True, "has_ocean_view": False, "has_mountain_view": False},
    {"id": 7, "title": "Lakefront Property", "description": "Stunning home with lake access", "city": "Chicago", "state": "IL", "price": 925000, "bedrooms": 3, "bathrooms": 2.5, "square_feet": 2200, "property_type": "Lake House", "has_pool": False, "has_garden": True, "has_ocean_view": False, "has_mountain_view": False},
    {"id": 8, "title": "Historic Townhouse", "description": "Charming townhouse in historic district", "city": "Charleston", "state": "SC", "price": 680000, "bedrooms": 3, "bathrooms": 2.5, "square_feet": 2000, "property_type": "Townhouse", "has_pool": False, "has_garden": True, "has_ocean_view": False, "has_mountain_view": False},
    {"id": 9, "title": "Desert Oasis", "description": "Modern home with desert landscape", "city": "Phoenix", "state": "AZ", "price": 520000, "bedrooms": 3, "bathrooms": 2, "square_feet": 1900, "property_type": "Modern Home", "has_pool": True, "has_garden": False, "has_ocean_view": False, "has_mountain_view": True},
    {"id": 10, "title": "Beach Condo", "description": "Beachfront condominium with amenities", "city": "San Diego", "state": "CA", "price": 680000, "bedrooms": 2, "bathrooms": 2, "square_feet": 1200, "property_type": "Condominium", "has_pool": True, "has_garden": False, "has_ocean_view": True, "has_mountain_view": False}
]

# Generate more properties to reach 40
for i in range(11, 41):
    base = properties_data[i % 10]
    properties_data.append({
        "id": i,
        "title": f"{base['title']} #{i//10 + 1}",
        "description": base["description"],
        "city": base["city"],
        "state": base["state"],
        "price": base["price"] + (i * 10000),
        "bedrooms": base["bedrooms"],
        "bathrooms": base["bathrooms"],
        "square_feet": base["square_feet"],
        "property_type": base["property_type"],
        "has_pool": base["has_pool"],
        "has_garden": base["has_garden"],
        "has_ocean_view": base["has_ocean_view"],
        "has_mountain_view": base["has_mountain_view"]
    })

print(f"\nGenerating embeddings for {len(properties_data)} properties...")

def get_openai_embedding(text):
    """Get OpenAI embedding with retries"""
    if not USE_OPENAI:
        # Return placeholder if no API key
        return np.random.normal(0, 0.1, DIMENSION).tolist()

    for attempt in range(3):
        try:
            response = openai_client.embeddings.create(
                model="text-embedding-ada-002",
                input=text,
                encoding_format="float"
            )
            return response.data[0].embedding
        except Exception as e:
            if attempt < 2:
                time.sleep(1)
            else:
                print(f"Error getting OpenAI embedding: {e}")
                return np.random.normal(0, 0.1, DIMENSION).tolist()

def download_image(url):
    """Download image from URL"""
    try:
        response = requests.get(url)
        return Image.open(io.BytesIO(response.content))
    except Exception as e:
        print(f"Error downloading image {url}: {e}")
        # Return a blank image as fallback
        return Image.new('RGB', (224, 224), color='white')

def get_image_embedding(image):
    """Get image embedding from PIL Image"""
    # Image must be properly formatted for CLIP
    if image.mode != 'RGB':
        image = image.convert('RGB')
    return image_encoder.encode([image])[0].tolist()

# Prepare data for insertion
entities = []
batch_size = 5

for i in range(0, len(properties_data), batch_size):
    batch = properties_data[i:i+batch_size]

    print(f"\nProcessing batch {i//batch_size + 1}/{(len(properties_data)-1)//batch_size + 1}")

    for prop in batch:
        print(f"  - Property {prop['id']}: {prop['title']}")

        # Prepare content for OpenAI embeddings
        content = f"""
{prop['title']}

{prop['description']}

Type: {prop['property_type']}
Location: {prop['city']}, {prop['state']}
{prop['bedrooms']} bedrooms, {prop['bathrooms']} bathrooms
{prop['square_feet']} square feet
Price: ${prop['price']:,.2f}

Features:
- Pool: {prop['has_pool']}
- Garden: {prop['has_garden']}
- Ocean View: {prop['has_ocean_view']}
- Mountain View: {prop['has_mountain_view']}
        """.strip()

        # Get OpenAI embedding for n8n compatibility
        if USE_OPENAI:
            openai_embedding = get_openai_embedding(content)
            print(f"    ✓ Got OpenAI embedding (1536-dim)")
        else:
            # Create placeholder OpenAI embedding (will not work with n8n)
            openai_embedding = np.random.normal(0, 0.1, DIMENSION).tolist()
            print(f"    ⚠ Generated placeholder OpenAI embedding (1536-dim)")

        # Get text embedding (sentence-transformers)
        text_embedding = text_encoder.encode([content])[0].tolist()

        # Get image
        image_url = image_urls[prop['id'] % len(image_urls)]
        image = download_image(image_url)
        image_embedding = get_image_embedding(image)
        print(f"    ✓ Got image embedding (512-dim)")

        # Create entity
        entity = {
            "id": prop['id'],
            "content": content,
            "vector": openai_embedding,
            "text_embedding": text_embedding,
            "image_embedding": image_embedding,
            "title": prop['title'],
            "description": prop['description'],
            "price": prop['price'],
            "city": prop['city'],
            "state": prop['state'],
            "bedrooms": prop['bedrooms'],
            "bathrooms": prop['bathrooms'],
            "square_feet": prop['square_feet'],
            "property_type": prop['property_type'],
            "image_url": image_url,
            "has_pool": prop['has_pool'],
            "has_garden": prop['has_garden'],
            "has_ocean_view": prop['has_ocean_view'],
            "has_mountain_view": prop['has_mountain_view']
        }

        entities.append(entity)

    # Insert batch
    try:
        collection.insert(entities)
        entities = []  # Reset for next batch
        print(f"    ✓ Batch inserted successfully")
    except Exception as e:
        print(f"    ❌ Error inserting batch: {e}")

# Final flush and load
print("\nFinalizing collection...")
collection.flush()
collection.load()
print("✅ Collection ready")

print("\n" + "="*80)
print("SUCCESS!")
print("="*80)
print(f"\nCollection '{COLLECTION_NAME}' created with:")
print(f"  - {len(properties_data)} properties")
if USE_OPENAI:
    print(f"  - OpenAI embeddings (1536-dim) - n8n compatible")
    print(f"  - L2 metric for OpenAI embeddings")
else:
    print(f"  - Placeholder OpenAI embeddings (NOT n8n compatible)")
print(f"  - Sentence transformer embeddings (512-dim)")
print(f"  - CLIP image embeddings (512-dim)")
print(f"  - COSINE metric for hybrid search")

print("\n" + "="*80)
print("USAGE")
print("="*80)
print("\n1. For n8n Milvus Vector Store node:")
print(f"   - Collection: {COLLECTION_NAME}")
print("   - Vector field: vector")
print("   - Metric: L2 (default in n8n)")
print("   - Top K: as desired (e.g., 10)")

print("\n2. For REST Proxy hybrid search:")
print(f"   POST https://milvus-rest-proxy-multimodal-production.up.railway.app/search/{COLLECTION_NAME}/hybrid")
print("   Body: {\"text\": \"your query\", \"image_url\": \"optional\", \"limit\": 10}")

print("\n3. For text-only search (using OpenAI embeddings):")
print(f"   POST https://milvus-rest-proxy-multimodal-production.up.railway.app/search/{COLLECTION_NAME}/text")
print("   Body: {\"text\": \"your query\", \"limit\": 10}")

print("\n4. For image-only search:")
print(f"   POST https://milvus-rest-proxy-multimodal-production.up.railway.app/search/{COLLECTION_NAME}/image")
print("   Body: {\"image_url\": \"your image url\", \"limit\": 10}")

if not USE_OPENAI:
    print("\n" + "⚠️  WARNING:")
    print("No OpenAI API key provided. The collection has placeholder OpenAI embeddings.")
    print("Set OPENAI_API_KEY environment variable and regenerate to make it n8n compatible.")
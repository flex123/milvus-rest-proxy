#!/usr/bin/env python3
"""Update property ID 2 to remove 'private pool' from description"""

import os
from pymilvus import connections, Collection
from sentence_transformers import SentenceTransformer
import numpy as np

# Connect to Milvus
connections.connect(
    alias="default",
    host="grpc-reverse-proxy-production-039b.up.railway.app",
    port=443,
    secure=True,
    timeout=30
)

print("Connected to Milvus")

# Load sentence transformer model
print("Loading text encoder model...")
text_encoder = SentenceTransformer("all-MiniLM-L6-v2")

# Get collection
collection = Collection("properties")
collection.load()

# Get current data for ID 2
results = collection.query(
    expr='id == 2',
    output_fields=[
        'id', 'title', 'description', 'price', 'city', 'state',
        'bedrooms', 'bathrooms', 'square_feet', 'property_type',
        'image_url', 'has_pool', 'has_garden', 'has_ocean_view',
        'has_mountain_view', 'neighborhood', 'address', 'zip_code',
        'year_built', 'text_embedding', 'image_embedding'
    ]
)

if not results:
    print("Property ID 2 not found")
    exit()

property_data = results[0]
print(f"\nCurrent description: {property_data['description']}")

# Create new description without "private pool"
new_description = "This penthouse offers 5 bedrooms, 5.6 bathrooms, 4374 sq ft of living space and built in 1974. Located in the prestigious neighborhood of Seattle, this property features high-end finishes and modern amenities throughout."

print(f"\nNew description: {new_description}")

# Prepare content for embedding (same format as original creation)
content = f"""
{property_data['title']}

{new_description}

Type: {property_data['property_type']}
Location: {property_data['city']}, {property_data['state']}
{property_data['bedrooms']} bedrooms, {property_data['bathrooms']} bathrooms
{property_data['square_feet']} square feet
Price: ${property_data['price']:,.2f}

Features:
- Pool: {property_data['has_pool']}
- Garden: {property_data['has_garden']}
- Ocean View: {property_data['has_ocean_view']}
- Mountain View: {property_data['has_mountain_view']}
""".strip()

# Generate new text embedding
print("\nGenerating new text embedding...")
new_text_embedding = text_encoder.encode([content])[0].tolist()

# Delete the old entity
print("\nDeleting old entity...")
collection.delete(expr='id == 2')

# Insert updated entity
print("Inserting updated entity...")
collection.insert([
    [2],  # id
    [new_text_embedding],  # text_embedding (updated)
    [property_data['image_embedding']],  # image_embedding (unchanged)
    [property_data['title']],  # title
    [new_description],  # description (updated)
    [property_data['price']],  # price
    [property_data['city']],  # city
    [property_data['state']],  # state
    [property_data['bedrooms']],  # bedrooms
    [property_data['bathrooms']],  # bathrooms
    [property_data['square_feet']],  # square_feet
    [property_data['property_type']],  # property_type
    [property_data['image_url']],  # image_url
    [bool(property_data['has_pool'])],  # has_pool (convert to bool)
    [bool(property_data['has_garden'])],  # has_garden
    [bool(property_data['has_ocean_view'])],  # has_ocean_view
    [bool(property_data['has_mountain_view'])],  # has_mountain_view
    [property_data['neighborhood']],  # neighborhood
    [property_data['address']],  # address
    [property_data['zip_code']],  # zip_code
    [property_data['year_built']]  # year_built
])

# Flush and reload
print("\nFinalizing...")
collection.flush()
collection.load()

print("\n✅ Property ID 2 updated successfully!")
print("Description no longer contains 'private pool'")
print("But has_pool field remains True")
#!/usr/bin/env python3
"""
Create new collection using Milvus 2.6.7 native hybrid search
Interactive version - prompts for configuration
"""

from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility
import random

# Dimensions
TEXT_DIM = 512  # CLIP text embedding dimension
IMAGE_DIM = 512  # CLIP image embedding dimension (same as text for CLIP)

def get_user_input():
    """Get configuration from user"""
    print("="*70)
    print("🔧 Milvus Native Hybrid Search Collection Setup")
    print("="*70)

    # Get Milvus connection details
    print("\n📡 Milvus Connection:")
    host = input("Enter Milvus host (default: grpc-reverse-proxy-production-039b.up.railway.app): ").strip()
    if not host:
        host = "grpc-reverse-proxy-production-039b.up.railway.app"

    port_str = input("Enter Milvus port (default: 443): ").strip()
    port = int(port_str) if port_str else 443

    secure = True  # Always use secure connection

    # Get collection details
    print("\n📦 Collection Configuration:")
    collection_name = input("Enter collection name (default: properties): ").strip()
    if not collection_name:
        collection_name = "properties"

    return {
        'host': host,
        'port': port,
        'secure': secure,
        'collection_name': collection_name
    }

def connect_to_milvus(config):
    """Connect to Milvus instance"""
    try:
        connections.connect(
            alias="default",
            host=config['host'],
            port=config['port'],
            secure=config['secure'],
            timeout=30
        )
        print(f"✅ Connected to Milvus at {config['host']}:{config['port']}")
        return True
    except Exception as e:
        print(f"❌ Failed to connect: {e}")
        return False

def create_collection_schema():
    """Create schema for native hybrid search with separate text and image vectors"""

    # Define fields
    fields = [
        # Primary key
        FieldSchema(
            name="id",
            dtype=DataType.INT64,
            is_primary=True,
            auto_id=False
        ),

        # Text embedding field (from sentence-transformers or CLIP)
        FieldSchema(
            name="text_embedding",
            dtype=DataType.FLOAT_VECTOR,
            dim=TEXT_DIM
        ),

        # Image embedding field (from CLIP)
        FieldSchema(
            name="image_embedding",
            dtype=DataType.FLOAT_VECTOR,
            dim=IMAGE_DIM
        ),

        # Property metadata (unchanged)
        FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=500),
        FieldSchema(name="description", dtype=DataType.VARCHAR, max_length=2000),
        FieldSchema(name="price", dtype=DataType.DOUBLE),
        FieldSchema(name="city", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="state", dtype=DataType.VARCHAR, max_length=50),
        FieldSchema(name="bedrooms", dtype=DataType.INT64),
        FieldSchema(name="bathrooms", dtype=DataType.DOUBLE),
        FieldSchema(name="square_feet", dtype=DataType.INT64),
        FieldSchema(name="property_type", dtype=DataType.VARCHAR, max_length=50),
        FieldSchema(name="has_pool", dtype=DataType.BOOL),
        FieldSchema(name="has_garden", dtype=DataType.BOOL),
        FieldSchema(name="has_ocean_view", dtype=DataType.BOOL),
        FieldSchema(name="has_mountain_view", dtype=DataType.BOOL),
        FieldSchema(name="neighborhood", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="address", dtype=DataType.VARCHAR, max_length=500),
        FieldSchema(name="zip_code", dtype=DataType.VARCHAR, max_length=10),
        FieldSchema(name="image_url", dtype=DataType.VARCHAR, max_length=1000),
        FieldSchema(name="year_built", dtype=DataType.INT64),
    ]

    # Create schema optimized for hybrid search
    schema = CollectionSchema(
        fields=fields,
        description="Property search with native Milvus 2.6.7 hybrid search",
        enable_dynamic_field=False
    )

    return schema

def create_collection(config):
    """Create and configure the collection with indexes for hybrid search"""

    collection_name = config['collection_name']

    # Check if collection already exists
    if utility.has_collection(collection_name):
        print(f"\n⚠️ Collection '{collection_name}' already exists")
        collection = Collection(collection_name)
        print(f"   Current entities: {collection.num_entities}")

        # Ask if user wants to drop and recreate
        response = input(f"\nDrop and recreate collection '{collection_name}'? (y/N): ").strip().lower()
        if response == 'y':
            utility.drop_collection(collection_name)
            print(f"🗑️ Dropped collection '{collection_name}'")
        else:
            print("   Keeping existing collection")
            return collection

    # Create schema
    schema = create_collection_schema()

    # Create collection
    collection = Collection(
        name=collection_name,
        schema=schema,
        using='default',
        shards_num=2  # Optimized for concurrent searches
    )

    print(f"✅ Created collection '{collection_name}'")

    # Create index for text embedding
    text_index_params = {
        "metric_type": "IP",  # Inner product for CLIP embeddings (better than L2)
        "index_type": "HNSW",  # HNSW for faster search on text embeddings
        "params": {"M": 16, "efConstruction": 256}
    }

    collection.create_index(
        field_name="text_embedding",
        index_params=text_index_params
    )
    print("✅ Created HNSW index on text_embedding field")

    # Create index for image embedding
    image_index_params = {
        "metric_type": "IP",  # Inner product for CLIP embeddings
        "index_type": "IVF_FLAT",  # IVF_FLAT for precise image similarity
        "params": {"nlist": 1024}
    }

    collection.create_index(
        field_name="image_embedding",
        index_params=image_index_params
    )
    print("✅ Created IVF_FLAT index on image_embedding field")

    return collection


def generate_and_insert_properties(collection, num_properties):
    """
    Generate properties with embeddings and insert into collection

    NOTE: This function now generates TRUE multi-modal embeddings:
    - Text embeddings: From title + description
    - Image embeddings: From actual image pixels (not just text!)
    This enables proper hybrid search that can find properties by visual content.
    """
    print("\n📦 Loading ML models for embedding generation...")
    from sentence_transformers import SentenceTransformer

    text_model = SentenceTransformer('all-MiniLM-L6-v2')
    clip_model = SentenceTransformer('clip-ViT-B-32')
    print("✅ Models loaded successfully")

    print(f"\n📝 Generating {num_properties} properties with embeddings...")

    # Import here to avoid circular import
    import numpy as np

    # Define property types and cities
    property_types = [
        "Single Family Home", "Condo", "Townhouse", "Apartment",
        "Luxury Villa", "Beach House", "Mountain Cabin", "Penthouse", "Studio"
    ]

    cities = [
        ("Miami", "FL"), ("Los Angeles", "CA"), ("New York", "NY"),
        ("Chicago", "IL"), ("Boston", "MA"), ("Seattle", "WA"),
        ("Austin", "TX"), ("Denver", "CO"), ("Portland", "OR"),
        ("Phoenix", "AZ"), ("San Francisco", "CA"), ("Malibu", "CA")
    ]

    # Sample image URLs
    image_urls = [
            "https://cdn.pixabay.com/photo/2016/12/30/07/59/kitchen-1940174_1280.jpg", #Modern Kitchen
            "https://cdn.pixabay.com/photo/2024/07/28/15/20/ai-generated-8927789_960_720.jpg", #Luxury Log Cabin
            "https://cdn.pixabay.com/photo/2014/12/06/12/47/fireplace-558985_960_720.jpg", #Cozy Fieplace
            "https://cdn.pixabay.com/photo/2020/09/14/17/16/barn-5571530_960_720.jpg", #Rustic Barn House
            "https://cdn.pixabay.com/photo/2020/02/04/02/59/tampa-4817193_1280.jpg", #Condo
            "https://cdn.pixabay.com/photo/2014/02/02/04/21/atlantic-city-256541_960_720.jpg", #Condo
            "https://cdn.pixabay.com/photo/2016/10/16/09/16/florida-1744691_960_720.jpg", #Beach House
            "https://cdn.pixabay.com/photo/2020/02/02/17/06/living-room-modern-tv-4813589_1280.jpg", #Modern Living Room
            "https://cdn.pixabay.com/photo/2023/12/11/06/20/living-room-8442806_1280.jpg", #Modern Living Room
            "https://cdn.pixabay.com/photo/2021/12/25/13/10/bathroom-6893077_1280.jpg", #Modern Bathroom
            "https://cdn.pixabay.com/photo/2017/10/24/13/13/home-2884521_1280.jpg", #Brick House
            "https://cdn.pixabay.com/photo/2023/04/09/12/57/master-bedroom-7911528_1280.jpg", #Luxury Bedroom
            "https://cdn.pixabay.com/photo/2017/07/14/18/47/barn-2504652_960_720.jpg", #Barn House
            "https://cdn.pixabay.com/photo/2018/03/15/15/51/house-3228636_960_720.jpg", #Barn House
            "https://cdn.pixabay.com/photo/2018/01/29/07/55/modern-minimalist-bathroom-3115450_960_720.jpg", #Modern Bathroom
            "https://cdn.pixabay.com/photo/2017/07/09/03/19/home-2486092_1280.jpg", #Nice kitchen
            "https://cdn.pixabay.com/photo/2018/06/13/13/32/yellow-building-3472814_1280.jpg", #Yellow house
            "https://cdn.pixabay.com/photo/2021/12/07/16/48/real-estate-6853680_1280.jpg", #Townhouse
            "https://cdn.pixabay.com/photo/2018/04/07/03/06/real-estate-3297625_1280.jpg", #Townhouse
            "https://cdn.pixabay.com/photo/2013/05/11/04/17/building-110285_1280.jpg", #Mansion
            "https://cdn.pixabay.com/photo/2018/05/26/14/46/manor-house-3431460_1280.jpg", #Mansion
            "https://cdn.pixabay.com/photo/2018/10/11/10/11/estonia-3739339_1280.jpg", #Mansion
            "https://cdn.pixabay.com/photo/2016/11/09/02/21/alfresco-1809841_1280.jpg", #Modern kitchen
            "https://cdn.pixabay.com/photo/2016/09/19/17/15/beautiful-home-1680787_960_720.jpg", #Beautiful lawn
            "https://cdn.pixabay.com/photo/2015/09/08/22/03/luggage-930804_1280.jpg", #Modern apartment with city view
            "https://cdn.pixabay.com/photo/2016/11/06/19/33/austria-1803887_960_720.jpg", #House with mountain view
            "https://cdn.pixabay.com/photo/2020/12/27/13/38/village-5863964_960_720.jpg", #House with mountain view
            "https://cdn.pixabay.com/photo/2023/01/09/17/36/lake-7707975_960_720.jpg", #House with lake view
            "https://cdn.pixabay.com/photo/2016/06/12/21/45/lake-1453079_960_720.jpg", #House with lake view
            "https://cdn.pixabay.com/photo/2019/09/15/14/23/fishermans-hut-4478431_1280.jpg", #Beach house
            "https://cdn.pixabay.com/photo/2016/09/20/11/28/new-home-1682323_1280.jpg", #Yellow house
            "https://cdn.pixabay.com/photo/2017/07/05/22/37/the-house-on-the-roof-2476243_1280.jpg", #Upside down house
            "https://cdn.pixabay.com/photo/2019/02/23/20/53/upside-down-house-4016494_1280.jpg",  #Upside down house
            "https://cdn.pixabay.com/photo/2017/10/07/07/41/old-house-2825712_1280.jpg", #Old house
            "https://cdn.pixabay.com/photo/2019/02/04/15/39/town-3974922_960_720.jpg", # Red roof houses
            "https://cdn.pixabay.com/photo/2024/03/28/16/59/house-8661359_1280.jpg", #Stone house
            "https://cdn.pixabay.com/photo/2018/11/05/16/36/old-house-3796405_960_720.jpg", #Stone house
            "https://cdn.pixabay.com/photo/2018/05/20/13/47/nordfriesland-3415793_960_720.jpg", #Thatched roof house
            "https://cdn.pixabay.com/photo/2017/10/05/19/29/home-2820617_960_720.jpg", #Thatched roof house
            "https://cdn.pixabay.com/photo/2020/04/17/12/28/pool-5055009_1280.jpg", #House with pool
            "https://cdn.pixabay.com/photo/2023/09/25/11/14/arra-luxury-8274729_1280.jpg", #House with pool
            "https://cdn.pixabay.com/photo/2018/03/06/04/16/swimming-pool-3202525_1280.jpg", #House with pool
            "https://cdn.pixabay.com/photo/2013/04/01/03/13/netherlands-98346_1280.jpg", #House with nice garden
            "https://cdn.pixabay.com/photo/2017/06/18/21/31/house-2417321_1280.jpg", #Modern house with garden
            "https://cdn.pixabay.com/photo/2017/07/03/21/35/house-2469067_1280.jpg", #Modern house with garden
            "https://cdn.pixabay.com/photo/2013/03/23/04/29/log-house-96085_1280.jpg", #Log house
            "https://cdn.pixabay.com/photo/2017/09/08/15/21/house-2729079_1280.jpg", #Apartment building
            "https://cdn.pixabay.com/photo/2013/04/10/04/28/ra-nijkerk-102385_1280.jpg", #House by the water
            "https://cdn.pixabay.com/photo/2016/08/15/21/44/house-1596545_960_720.jpg", #House by the canal
            "https://cdn.pixabay.com/photo/2020/08/06/22/03/pool-5469283_960_720.jpg" #House with pool
        ]

    entities = []

    for i in range(num_properties):
        # Basic property info
        property_type = random.choice(property_types)
        city, state = random.choice(cities)
        bedrooms = random.randint(1, 6)
        bathrooms = round(bedrooms * random.uniform(0.8, 1.5), 1)
        square_feet = random.randint(800, 5000)
        year_built = random.randint(1950, 2024)

        # Price based on location and size
        base_price = {
            "FL": 400000, "CA": 800000, "NY": 750000, "IL": 350000,
            "MA": 600000, "WA": 650000, "TX": 350000, "CO": 500000,
            "OR": 450000, "AZ": 400000
        }.get(state, 400000)

        price_adjustment = (square_feet / 1000 - 1) * 100000
        year_factor = 1 + ((year_built - 2000) / 100) * 0.1
        property_multiplier = {
            "Single Family Home": 1.0,
            "Condo": 0.7,
            "Townhouse": 0.85,
            "Apartment": 0.6,
            "Luxury Villa": 2.5,
            "Beach House": 1.8,
            "Mountain Cabin": 0.9,
            "Penthouse": 2.0,
            "Studio": 0.4
        }.get(property_type, 1.0)

        price = max(150000, int((base_price + price_adjustment) * year_factor * property_multiplier))

        # Random features
        has_pool = random.choice([True, False]) if property_type in ["Single Family Home", "Luxury Villa", "Beach House", "Penthouse"] else False
        has_garden = random.choice([True, False]) if "Square Feet" in str(square_feet) else False
        has_ocean_view = random.choice([True, False]) if city in ["Miami Beach", "Malibu", "Santa Monica"] else False
        has_mountain_view = random.choice([True, False]) if city in ["Denver", "Aspen", "Seattle"] else False

        # Generate title and description
        title_templates = [
            f"{property_type} in {city}",
            f"Beautiful {property_type.lower()} with {bedrooms} bedrooms",
            f"Modern {property_type.lower()} in {city}'s {random.choice(['downtown', 'uptown', 'historic district', 'suburban area'])}",
            f"Spacious {property_type.lower()} with {random.choice(['stunning', 'beautiful', 'charming', 'elegant'])} views",
            f"{property_type} near {random.choice(['schools', 'parks', 'shopping', 'transit'])}"
        ]

        title = random.choice(title_templates)

        # Generate description with features
        features = []
        if bedrooms > 1:
            features.append(f"{bedrooms} bedrooms")
        if bathrooms >= 2:
            features.append(f"{bathrooms} bathrooms")
        if square_feet > 2000:
            features.append(f"{square_feet} sq ft of living space")
        if has_pool:
            features.append("private pool")
        if has_garden:
            features.append("beautiful garden")
        if has_ocean_view:
            features.append("ocean views")
        if has_mountain_view:
            features.append("mountain views")

        features.append(f"built in {year_built}")

        description = f"This {property_type.lower()} offers " + ", ".join(features[:-1]) + f" and {features[-1]}. "

        if property_type in ["Luxury Villa", "Penthouse"]:
            description += f"Located in the prestigious {random.choice(['area', 'neighborhood', 'community'])} of {city}, "
            description += f"this property features high-end finishes and modern amenities throughout."
        else:
            description += f"Perfect for {random.choice(['families', 'professionals', 'retirees', 'students'])} "
            description += f"looking for {random.choice(['comfortable living', 'a great investment', 'their dream home'])}."

        # Use unique image for each property (no random selection)
        image_url = image_urls[i % len(image_urls)]

        # Create address and zip
        street_numbers = [random.randint(100, 9999) for _ in range(3)]
        street_names = ["Oak", "Maple", "Pine", "Cedar", "Elm", "Park", "Main", "Washington", "Lincoln", "Jefferson"]
        street_types = ["St", "Ave", "Dr", "Ln", "Blvd", "Ct", "Pl", "Rd"]
        address = f"{random.choice(street_numbers)} {random.choice(street_names)} {random.choice(street_types)}"

        zip_code = f"{random.randint(10000, 99999)}"

        print(f"   Processing {i+1}/{num_properties}: {title[:40]}...")

        # Generate embeddings
        text_for_embedding = f"{title} {description}"
        text_embedding = text_model.encode(text_for_embedding)

        # Ensure 512 dimensions
        if text_embedding.shape[0] > 512:
            text_embedding = text_embedding[:512]
        elif text_embedding.shape[0] < 512:
            text_embedding = np.pad(text_embedding, (0, 512 - text_embedding.shape[0]))
        text_embedding = text_embedding.astype(np.float32)

        # Generate image embedding from the actual image
        try:
            import requests
            from PIL import Image
            from io import BytesIO

            # Download and process the image
            response = requests.get(image_url, timeout=10)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content))
            if image.mode != 'RGB':
                image = image.convert('RGB')

            # Use CLIP to encode the actual image
            image_embedding = clip_model.encode(image)
            image_embedding = image_embedding.astype(np.float32)
        except Exception as e:
            print(f"      Warning: Failed to process image {image_url}: {e}")
            # Fallback to text-based embedding
            image_embedding = clip_model.encode([title])[0]
            image_embedding = image_embedding.astype(np.float32)

        # Create entity
        entity = {
            "id": i + 1,
            "title": title,
            "description": description,
            "price": price,
            "city": city,
            "state": state,
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "square_feet": square_feet,
            "property_type": property_type,
            "has_pool": has_pool,
            "has_garden": has_garden,
            "has_ocean_view": has_ocean_view,
            "has_mountain_view": has_mountain_view,
            "neighborhood": random.choice(["Downtown", "Uptown", "Historic District", "Suburbs", "Waterfront"]),
            "address": address,
            "zip_code": zip_code,
            "image_url": image_url,
            "year_built": year_built,
            "text_embedding": text_embedding,
            "image_embedding": image_embedding
        }

        entities.append(entity)

        # Insert in batches of 10 for efficiency
        if len(entities) >= 10 or (i == num_properties - 1):
            try:
                result = collection.insert(entities[-(len(entities) % 10 or 10):])
                print(f"      Inserted batch, IDs: {result.primary_keys[-(len(entities) % 10 or 10):]}")
            except Exception as e:
                print(f"   ❌ Failed to insert batch: {e}")

    # Flush to ensure data is persisted
    try:
        collection.flush()
        print(f"\n✅ Successfully generated and inserted {num_properties} properties!")

        # Refresh collection stats
        collection.load()
        print(f"📊 Total entities in collection: {collection.num_entities}")

    except Exception as e:
        print(f"\n❌ Failed during data insertion: {e}")
        import traceback
        traceback.print_exc()

def main():

    # Get configuration from user
    config = get_user_input()

    print("\n" + "="*70)
    print("🚀 Creating Native Hybrid Search Collection")
    print(f"📊 Using Milvus v2.6.7 hybrid_search() API")
    print(f"📦 Collection: {config['collection_name']}")
    print(f"📝 Text vector: {TEXT_DIM} dimensions")
    print(f"🖼️ Image vector: {IMAGE_DIM} dimensions")
    print("="*70)

    # Connect to Milvus
    if not connect_to_milvus(config):
        print("\n❌ Failed to connect to Milvus. Please check your connection details and try again.")
        return

    # Create collection
    collection = create_collection(config)

    # Ask if user wants to generate properties
    generate_data = input("\nGenerate sample properties for the collection? (y/N): ").strip().lower()

    if generate_data == 'y':
        num_str = input("How many properties to generate? (default: 50): ").strip()
        num_properties = int(num_str) if num_str else 50

        # Generate properties with embeddings
        print("\n⚠️  This requires downloading ML models and may take several minutes.")
        proceed = input("Continue with embedding generation? (y/N): ").strip().lower()
        if proceed == 'y':
            print("   Starting embedding generation...")
            generate_and_insert_properties(collection, num_properties)

    # Show collection info
    print("\n" + "="*70)
    print("✅ Collection Setup Completed!")
    print("="*70)

    collection.load()
    print(f"\n📋 Collection Details:")
    print(f"   Name: {collection.name}")
    print(f"   Description: {collection.description}")
    print(f"   Fields: {len(collection.schema.fields)}")
    print(f"   Entities: {collection.num_entities}")

    print("\n🏷️ Vector Fields:")
    for field in collection.schema.fields:
        if field.dtype == DataType.FLOAT_VECTOR:
            print(f"   - {field.name}: {field.dtype} (dim={field.dim})")

    print("\n🔍 Search Capabilities:")
    print("   ✅ Native hybrid_search() with multiple vectors")
    print("   ✅ RRFRanker for balanced fusion")
    print("   ✅ WeightedReranker for custom text/image weights")
    print("   ✅ Independent filtering per vector field")

    print("\n💡 Usage Example:")
    print("""
    from pymilvus import AnnSearchRequest, RRFReranker

    # Create search requests
    req1 = AnnSearchRequest(data=[text_vec], anns_field="text_embedding", ...)
    req2 = AnnSearchRequest(data=[image_vec], anns_field="image_embedding", ...)

    # Hybrid search with 70% text, 30% image
    rerank = RRFReranker(weights=[0.7, 0.3])
    results = collection.hybrid_search([req1, req2], rerank=rerank, limit=10)
    """)

if __name__ == "__main__":
    main()
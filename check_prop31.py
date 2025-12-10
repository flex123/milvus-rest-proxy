from pymilvus import connections, Collection

# Connect to Milvus
connections.connect(
    alias='default',
    host='grpc-reverse-proxy-production-039b.up.railway.app',
    port=443,
    secure=True,
    timeout=30
)

# Get collection
collection = Collection('properties')
collection.load()

# Get property ID 31
results = collection.query(
    expr='id == 31',
    output_fields=['id', 'title', 'description', 'has_pool', 'image_url', 'price', 'city', 'property_type']
)

if results:
    prop = results[0]
    print('Property ID 31:')
    print('  Title:', prop['title'])
    print('  City:', prop['city'])
    print('  Type:', prop['property_type'])
    print('  Has pool:', prop['has_pool'])
    print('  Price: ${:,.0f}'.format(prop['price']))
    print('  Image URL:', prop['image_url'])
    print()
    print('Description:', prop['description'])

    # Check if description mentions pool
    desc = prop.get('description', '').lower()
    if 'pool' in desc:
        print()
        print('✅ Description mentions "pool"')
    else:
        print()
        print('❌ Description does NOT mention "pool"')

    # Check if this makes sense for the search
    if not prop['has_pool'] and 'pool' not in desc:
        print()
        print('⚠️  This should NOT rank high for "A pool." search')
        print('   - No pool in metadata')
        print('   - No pool in description')
        print('   - Image must be visually similar to pool-related queries')
        print('   - Image filename:', prop['image_url'].split('/')[-1])
else:
    print('Property ID 31 not found')
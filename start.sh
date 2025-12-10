#!/bin/bash

echo "Starting Milvus REST Proxy..."

# Check if MILVUS_HOST is set
if [ -z "$MILVUS_HOST" ]; then
    echo "ERROR: MILVUS_HOST environment variable is required!"
    echo ""
    echo "Please set MILVUS_HOST in one of the following ways:"
    echo "1. Railway Variables tab: MILVUS_HOST=your-milvus-host"
    echo "2. Railway Start Command: MILVUS_HOST=your-milvus-host ./start.sh"
    echo ""
    echo "Examples:"
    echo "  MILVUS_HOST=localhost"
    echo "  MILVUS_HOST=grpc-reverse-proxy-production-039b.up.railway.app"
    echo "  MILVUS_HOST=in01-xxxxxxxx.aws-us-west-2.vectordb.zilliz.com"
    exit 1
fi

echo "MILVUS_HOST: $MILVUS_HOST"
echo "MILVUS_PORT: ${MILVUS_PORT:-443}"

# Start the application
exec python3 rest-proxy-multimodal.py
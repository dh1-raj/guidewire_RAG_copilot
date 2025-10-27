from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, Distance, VectorParams
from qdrant_client.http.exceptions import UnexpectedResponse
import logging

logger = logging.getLogger(__name__)

def get_qdrant_client(url, api_key):
    """Initialize Qdrant client with longer timeout."""
    logger.info(f"Initializing Qdrant client with URL: {url}")
    
    # Suppress version compatibility warnings
    import warnings
    warnings.filterwarnings('ignore', message='.*Qdrant client version.*incompatible.*')
    
    if api_key:
        client = QdrantClient(url=url, api_key=api_key, timeout=120)
        logger.info("Qdrant client initialized with API key and 120s timeout")
    else:
        client = QdrantClient(url=url, timeout=120)
        logger.info("Qdrant client initialized without API key and 120s timeout")
    return client

def collection_exists(client, collection_name):
    """Check if collection exists."""
    logger.debug(f"Checking if collection '{collection_name}' exists")
    try:
        client.get_collection(collection_name)
        logger.debug(f"Collection '{collection_name}' exists")
        return True
    except UnexpectedResponse:
        logger.debug(f"Collection '{collection_name}' does not exist")
        return False

def create_collection(client, collection_name, vector_size):
    """Create or recreate collection."""
    logger.info(f"Creating collection '{collection_name}' with vector size {vector_size}")
    if collection_exists(client, collection_name):
        logger.info(f"Collection '{collection_name}' already exists, deleting it")
        client.delete_collection(collection_name)
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
    )
    logger.info(f"Collection '{collection_name}' created successfully")

def upload_documents(client, collection_name, docs, embeddings):
    """Upload documents and embeddings to Qdrant."""
    points = [
        PointStruct(id=i, vector=embeddings[i], payload={"text": docs[i]})
        for i in range(len(docs))
    ]
    client.upsert(collection_name=collection_name, points=points)

def upload_documents_with_metadata(client, collection_name, chunk_objs, embeddings):
    """Upload documents with metadata to Qdrant in batches to avoid timeouts."""
    total_docs = len(chunk_objs)
    logger.info(f"Uploading {total_docs} documents to collection '{collection_name}'")
    
    # Upload in batches of 100 to avoid timeout
    batch_size = 100
    total_batches = (total_docs + batch_size - 1) // batch_size
    
    for batch_num in range(total_batches):
        start_idx = batch_num * batch_size
        end_idx = min((batch_num + 1) * batch_size, total_docs)
        
        logger.info(f"Uploading batch {batch_num + 1}/{total_batches} ({end_idx - start_idx} points)")
        
        batch_points = [
            PointStruct(
                id=i, 
                vector=embeddings[i], 
                payload={
                    "text": chunk_objs[i]["text"],
                    "source": chunk_objs[i]["metadata"]["source"],
                    "chunk_index": chunk_objs[i]["metadata"]["chunk_index"],
                    "length": chunk_objs[i]["metadata"]["length"]
                }
            )
            for i in range(start_idx, end_idx)
        ]
        
        client.upsert(collection_name=collection_name, points=batch_points)
    
    logger.info(f"Successfully uploaded {total_docs} points to Qdrant in {total_batches} batch(es)")


def search_documents(client, collection_name, query_embedding, top_k=3):
    """Search for similar documents. Returns only text (legacy)."""
    results_with_metadata = search_documents_with_metadata(client, collection_name, query_embedding, top_k)
    return [r['text'] for r in results_with_metadata]

def search_documents_with_metadata(client, collection_name, query_embedding, top_k=3):
    """Search for similar documents and return full metadata."""
    logger.info(f"Searching collection '{collection_name}' for top {top_k} documents")
    if not collection_exists(client, collection_name):
        logger.warning(f"Collection '{collection_name}' does not exist")
        return []
    
    results = client.search(
        collection_name=collection_name,
        query_vector=query_embedding,
        limit=top_k
    )
    
    logger.info(f"Found {len(results)} relevant documents")
    
    formatted_results = []
    for i, r in enumerate(results, 1):
        result_data = {
            'text': r.payload['text'],
            'source': r.payload.get('source', 'Unknown'),
            'chunk_index': r.payload.get('chunk_index', 0),
            'page_number': r.payload.get('page_number'),
            'score': round(r.score, 4)
        }
        formatted_results.append(result_data)
        
        page_info = f"page {result_data['page_number']}" if result_data['page_number'] else f"chunk {result_data['chunk_index']}"
        logger.debug(f"Result {i}: score={r.score:.4f}, source={result_data['source']}, {page_info}")
    
    return formatted_results

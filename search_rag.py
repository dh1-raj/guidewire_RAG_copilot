#!/usr/bin/env python3
"""Quick search script for RAG knowledge base"""

import sys
from qdrant_utils import get_qdrant_client, search_documents_with_metadata
from embedder import get_embedding
import os
from dotenv import load_dotenv

load_dotenv()

def search_knowledge_base(query, top_k=5):
    """Search the RAG knowledge base"""
    client = get_qdrant_client(os.getenv("QDRANT_URL"), None)
    
    print(f"🔍 Searching for: '{query}'\n")
    
    query_embedding = get_embedding(query, os.getenv("OPENAI_API_KEY"))
    results = search_documents_with_metadata(client, "reference_docs", query_embedding, top_k)
    
    for i, r in enumerate(results, 1):
        page_info = f"Page {r['page_number']}" if r['page_number'] else f"Chunk {r['chunk_index']}"
        print(f"\n{'='*70}")
        print(f"Result {i}: {r['source']} - {page_info} ({r['score']*100:.0f}% match)")
        print(f"{'='*70}")
        print(r['text'])
        print()

if __name__ == "__main__":
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "event driven architecture author"
    search_knowledge_base(query)

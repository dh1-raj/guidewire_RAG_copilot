#!/usr/bin/env python3
"""
Interactive RAG CLI - Query your knowledge base without typing the full command
"""

import sys
import os
from dotenv import load_dotenv

load_dotenv()

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from embedder import get_embedding
from qdrant_utils import get_qdrant_client, search_documents_with_metadata

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
COLLECTION_NAME = "reference_docs"


def search_interactive():
    """Interactive search mode - keeps running until user quits"""
    print("=" * 70)
    print("🤖 RAG Knowledge Base - Interactive Search")
    print("=" * 70)
    print("\nType your questions and press Enter. Type 'quit' or 'exit' to stop.\n")
    
    # Initialize connection once
    try:
        client = get_qdrant_client(QDRANT_URL, QDRANT_API_KEY)
        print("✓ Connected to Qdrant\n")
    except Exception as e:
        print(f"❌ Error connecting to Qdrant: {e}")
        return
    
    while True:
        try:
            # Get query from user
            query = input("📝 Your question: ").strip()
            
            # Check for exit commands
            if query.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            if not query:
                continue
            
            # Search
            print(f"\n🔍 Searching for: '{query}'")
            print()
            
            # Generate embedding
            query_embedding = get_embedding(query, OPENAI_API_KEY)
            
            # Search documents
            results = search_documents_with_metadata(
                client, 
                COLLECTION_NAME, 
                query_embedding, 
                top_k=5
            )
            
            if not results:
                print("❌ No results found. Try a different query.\n")
                continue
            
            # Display results
            for i, result in enumerate(results, 1):
                relevance_pct = int(result['score'] * 100)
                
                print("=" * 70)
                print(f"Result {i}: {result['source']} - Chunk {result['chunk_index']} ({relevance_pct}% match)")
                print("=" * 70)
                print(result['text'])
                print()
            
            print("=" * 70)
            print()
            
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")


def search_once(query):
    """Single search mode - run once and exit"""
    print(f"🔍 Searching for: '{query}'\n")
    
    try:
        client = get_qdrant_client(QDRANT_URL, QDRANT_API_KEY)
        query_embedding = get_embedding(query, OPENAI_API_KEY)
        results = search_documents_with_metadata(
            client, 
            COLLECTION_NAME, 
            query_embedding, 
            top_k=5
        )
        
        if not results:
            print("❌ No results found.")
            return
        
        for i, result in enumerate(results, 1):
            relevance_pct = int(result['score'] * 100)
            
            print("=" * 70)
            print(f"Result {i}: {result['source']} - Chunk {result['chunk_index']} ({relevance_pct}% match)")
            print("=" * 70)
            print(result['text'])
            print()
        
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Single query mode
        query = " ".join(sys.argv[1:])
        search_once(query)
    else:
        # Interactive mode
        search_interactive()

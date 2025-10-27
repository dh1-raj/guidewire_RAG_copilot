# Model Context Protocol (MCP) Server for GitHub Copilot Integration
# This allows GitHub Copilot to query your RAG knowledge base directly

import asyncio
import json
from typing import Any
import os
from dotenv import load_dotenv

# MCP Server imports
try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp import types
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    print("⚠️  MCP not installed. Run: pip install mcp")

# Your existing RAG components
from qdrant_utils import get_qdrant_client, search_documents_with_metadata
from embedder import get_embedding

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
COLLECTION_NAME = "reference_docs"

# Initialize Qdrant client (reused across requests)
qdrant_client = None

def get_client():
    global qdrant_client
    if qdrant_client is None:
        qdrant_client = get_qdrant_client(QDRANT_URL, QDRANT_API_KEY)
    return qdrant_client

# Create MCP server
if MCP_AVAILABLE:
    server = Server("rag-knowledge-base")
    
    @server.list_tools()
    async def list_tools() -> list[types.Tool]:
        """
        List available tools that GitHub Copilot can use.
        These tools will appear in Copilot's context menu.
        """
        return [
            types.Tool(
                name="search_knowledge_base",
                description="""
                Search the RAG knowledge base for relevant documentation, code examples, 
                and API references. Use this to find context from uploaded documentation 
                before generating code.
                
                Best for:
                - Finding API usage examples
                - Looking up configuration patterns
                - Retrieving code snippets from documentation
                - Getting context about specific technologies
                """,
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Natural language query describing what you're looking for"
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Number of results to return (default: 5, max: 10)",
                            "default": 5
                        }
                    },
                    "required": ["query"]
                }
            ),
            types.Tool(
                name="get_file_contents",
                description="""
                Retrieve full contents of a specific file from the knowledge base.
                Use when you need complete context from a particular document.
                """,
                inputSchema={
                    "type": "object",
                    "properties": {
                        "filename": {
                            "type": "string",
                            "description": "Name of the file to retrieve"
                        }
                    },
                    "required": ["filename"]
                }
            )
        ]
    
    @server.call_tool()
    async def call_tool(name: str, arguments: Any) -> list[types.TextContent]:
        """
        Handle tool calls from GitHub Copilot.
        When Copilot needs context, it calls these tools.
        """
        
        if name == "search_knowledge_base":
            query = arguments.get("query", "")
            top_k = arguments.get("top_k", 5)
            
            try:
                # Generate embedding for query
                query_embedding = get_embedding(query, OPENAI_API_KEY)
                
                # Search Qdrant
                client = get_client()
                results = search_documents_with_metadata(
                    client, 
                    COLLECTION_NAME, 
                    query_embedding, 
                    top_k
                )
                
                if not results:
                    return [types.TextContent(
                        type="text",
                        text="No relevant documentation found in knowledge base."
                    )]
                
                # Format results for Copilot
                formatted_results = []
                formatted_results.append(f"Found {len(results)} relevant sources:\n")
                
                for i, result in enumerate(results, 1):
                    page_info = f"Page {result['page_number']}" if result['page_number'] else f"Section {result['chunk_index']}"
                    relevance = f"{result['score']:.0%}"
                    
                    formatted_results.append(f"\n{'='*60}")
                    formatted_results.append(f"Source {i}: {result['source']} - {page_info} ({relevance} match)")
                    formatted_results.append(f"{'='*60}")
                    formatted_results.append(result['text'])
                    formatted_results.append("")
                
                return [types.TextContent(
                    type="text",
                    text="\n".join(formatted_results)
                )]
                
            except Exception as e:
                return [types.TextContent(
                    type="text",
                    text=f"Error searching knowledge base: {str(e)}"
                )]
        
        elif name == "get_file_contents":
            filename = arguments.get("filename", "")
            
            try:
                client = get_client()
                
                # Search for all chunks from this file
                # We use a broad query and filter by source
                query_embedding = get_embedding("content", OPENAI_API_KEY)
                results = search_documents_with_metadata(
                    client,
                    COLLECTION_NAME,
                    query_embedding,
                    limit=1000  # Get many results
                )
                
                # Filter by filename
                file_chunks = [r for r in results if r['source'] == filename]
                
                if not file_chunks:
                    return [types.TextContent(
                        type="text",
                        text=f"File '{filename}' not found in knowledge base."
                    )]
                
                # Sort by chunk index and combine
                file_chunks.sort(key=lambda x: x['chunk_index'])
                full_content = "\n\n".join(chunk['text'] for chunk in file_chunks)
                
                return [types.TextContent(
                    type="text",
                    text=f"Contents of {filename}:\n\n{full_content}"
                )]
                
            except Exception as e:
                return [types.TextContent(
                    type="text",
                    text=f"Error retrieving file: {str(e)}"
                )]
        
        else:
            return [types.TextContent(
                type="text",
                text=f"Unknown tool: {name}"
            )]

async def main():
    """
    Run the MCP server.
    GitHub Copilot will connect to this server to access your knowledge base.
    """
    if not MCP_AVAILABLE:
        print("❌ MCP not installed. Install with: pip install mcp")
        return
    
    print("🚀 Starting RAG Knowledge Base MCP Server...")
    print(f"📚 Collection: {COLLECTION_NAME}")
    print(f"🔗 Qdrant URL: {QDRANT_URL}")
    print("✅ Server ready for GitHub Copilot connections")
    print("\nConfigure in VS Code settings.json:")
    print('''
{
  "github.copilot.advanced": {
    "debug.overrideEngine": "gpt-4",
    "contextProviders": {
      "mcp": {
        "command": "python",
        "args": ["mcp_server.py"],
        "cwd": "/Users/dhiraj/Github copilot builds/Project_1"
      }
    }
  }
}
    ''')
    
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )

if __name__ == "__main__":
    if MCP_AVAILABLE:
        asyncio.run(main())
    else:
        print("\n📦 Install MCP support:")
        print("pip install mcp")
        print("\nThen run: python mcp_server.py")

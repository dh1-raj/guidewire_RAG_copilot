from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from embedder import extract_text_with_pages, get_embedding, get_embeddings_batch
from document_processor import clean_text, chunk_by_sentences, create_chunk_with_metadata
from qdrant_utils import get_qdrant_client, create_collection, upload_documents_with_metadata, search_documents_with_metadata
from generator import generate_code, generate_code_stream

from prompt import build_prompt
import os
import logging
import time
import json
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'rag_pipeline_{datetime.now().strftime("%Y%m%d")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
COLLECTION_NAME = "reference_docs"

@app.post("/upload")
async def upload_docs(files: list[UploadFile] = File(...)):
    """
    Complete RAG pipeline: Extract → Clean → Chunk → Embed → Store
    """
    pipeline_start = time.time()
    logger.info(f"========== Starting document upload pipeline ==========")
    logger.info(f"Received {len(files)} files for processing")
    
    if not files:
        logger.warning("No files provided in request")
        return {"status": "error", "message": "No files provided"}
    
    client_start = time.time()
    client = get_qdrant_client(QDRANT_URL, QDRANT_API_KEY)
    client_time = time.time() - client_start
    logger.info(f"Qdrant client initialized successfully (took {client_time:.2f}s)")
    
    all_chunk_objs = []
    all_embeddings = []
    progress_log = []
    timing_info = {}
    
    for file_idx, file in enumerate(files, 1):
        file_start = time.time()
        logger.info(f"[{file_idx}/{len(files)}] Processing file: {file.filename}")
        progress_log.append(f"[{file_idx}/{len(files)}] Processing: {file.filename}")
        
        # Read file
        read_start = time.time()
        file_bytes = await file.read()
        read_time = time.time() - read_start
        file_size_mb = len(file_bytes) / (1024 * 1024)
        logger.info(f"  → File size: {file_size_mb:.2f} MB (read took {read_time:.2f}s)")
        progress_log.append(f"  → File size: {file_size_mb:.2f} MB")
        
        # Step 1: Extract with page information
        extract_start = time.time()
        logger.info(f"  → Step 1: Extracting text from {file.filename}")
        progress_log.append(f"  → Extracting text with page tracking...")
        raw_text, page_texts = extract_text_with_pages(file_bytes, file.filename)
        extract_time = time.time() - extract_start
        
        if not raw_text:
            logger.warning(f"  ✗ No text extracted from {file.filename}, skipping")
            progress_log.append(f"  ✗ Failed to extract text, skipping")
            continue
        
        logger.info(f"  ✓ Extracted {len(raw_text)} characters from {len(page_texts)} pages (took {extract_time:.2f}s)")
        progress_log.append(f"  ✓ Extracted {len(raw_text)} characters from {len(page_texts)} pages in {extract_time:.2f}s")
        
        # Step 2: Process each page separately for better citation tracking
        chunk_start = time.time()
        logger.info(f"  → Step 2-3: Processing {len(page_texts)} pages into chunks")
        progress_log.append(f"  → Processing pages into chunks...")
        
        page_chunks = []
        for page_num, page_text in page_texts:
            # Clean page text
            cleaned_page = clean_text(page_text)
            if not cleaned_page:
                continue
            
            # Chunk page with semantic boundaries
            chunks = chunk_by_sentences(cleaned_page, chunk_size=500, overlap=50)
            for chunk in chunks:
                page_chunks.append((page_num, chunk))
        
        chunk_time = time.time() - chunk_start
        
        if not page_chunks:
            logger.warning(f"  ✗ No chunks created from {file.filename}, skipping")
            progress_log.append(f"  ✗ Failed to create chunks, skipping")
            continue
        
        logger.info(f"  ✓ Created {len(page_chunks)} chunks from {len(page_texts)} pages (took {chunk_time:.2f}s)")
        progress_log.append(f"  ✓ Created {len(page_chunks)} chunks in {chunk_time:.2f}s")
        
        # Step 4-5: Create chunks with metadata & Generate embeddings IN BATCH
        logger.info(f"  → Step 4-5: Adding metadata and generating embeddings for {len(page_chunks)} chunks")
        progress_log.append(f"  → Batch embedding {len(page_chunks)} chunks (OPTIMIZED)...")
        
        embed_start = time.time()
        
        # Create chunk objects with metadata
        file_chunk_objs = []
        chunk_texts = []
        for idx, (page_num, chunk) in enumerate(page_chunks):
            chunk_obj = create_chunk_with_metadata(chunk, file.filename, idx, page_num)
            file_chunk_objs.append(chunk_obj)
            chunk_texts.append(chunk)
        
        # BATCH EMBEDDING - This is the key optimization!
        # Instead of 300 sequential API calls, we make 3 batch calls
        logger.info(f"  → Generating {len(chunk_texts)} embeddings in batches (100 per call)")
        file_embeddings = get_embeddings_batch(chunk_texts, OPENAI_API_KEY, batch_size=100)
        
        # Add to global lists
        all_chunk_objs.extend(file_chunk_objs)
        all_embeddings.extend(file_embeddings)
        
        embed_time = time.time() - embed_start
        embeddings_per_sec = len(page_chunks) / embed_time if embed_time > 0 else 0
        logger.info(f"  ✓ Generated {len(page_chunks)} embeddings (took {embed_time:.2f}s, {embeddings_per_sec:.1f} embeddings/sec)")
        progress_log.append(f"  ✓ Batch generated {len(page_chunks)} embeddings in {embed_time:.2f}s ({embeddings_per_sec:.1f}/sec)")

        
        file_total_time = time.time() - file_start
        logger.info(f"  → Total time for {file.filename}: {file_total_time:.2f}s")
        progress_log.append(f"  → Total: {file_total_time:.2f}s")
        
        # Store timing for this file
        timing_info[file.filename] = {
            "file_size_mb": file_size_mb,
            "read_time": read_time,
            "extract_time": extract_time,
            "chunk_time": chunk_time,
            "embed_time": embed_time,
            "total_time": file_total_time,
            "chunks_created": len(page_chunks),
            "pages_extracted": len(page_texts),
            "chars_extracted": len(raw_text)
        }
    
    if not all_embeddings:
        logger.error("No valid content found in any uploaded files")
        return {"status": "error", "message": "No valid content found in uploaded files", "progress": progress_log}
    
    # Step 6: Store in vector database
    store_start = time.time()
    logger.info(f"→ Step 6: Storing {len(all_chunk_objs)} chunks in Qdrant")
    progress_log.append(f"→ Storing {len(all_chunk_objs)} chunks in Qdrant...")
    create_collection(client, COLLECTION_NAME, len(all_embeddings[0]))
    upload_documents_with_metadata(client, COLLECTION_NAME, all_chunk_objs, all_embeddings)
    store_time = time.time() - store_start
    logger.info(f"✓ Successfully uploaded {len(all_chunk_objs)} chunks to collection '{COLLECTION_NAME}' (took {store_time:.2f}s)")
    progress_log.append(f"✓ Successfully stored all chunks in {store_time:.2f}s!")
    
    pipeline_total_time = time.time() - pipeline_start
    logger.info(f"========== Upload pipeline completed successfully in {pipeline_total_time:.2f}s ==========")
    
    # Add timing summary
    timing_summary = {
        "total_pipeline_time": round(pipeline_total_time, 2),
        "qdrant_client_init": round(client_time, 2),
        "vector_store_time": round(store_time, 2),
        "files": timing_info
    }
    
    progress_log.append(f"\n⏱️ Total pipeline time: {pipeline_total_time:.2f}s")
    
    return {
        "status": "success", 
        "chunks": len(all_chunk_objs),
        "files_processed": len(files),
        "progress": progress_log,
        "timing": timing_summary
    }


@app.post("/generate")
async def generate(query: str = Form(...), top_k: int = Form(5), conversation_history: str = Form("")):
    logger.info(f"========== Starting code generation pipeline ==========")
    logger.info(f"Query: {query[:100]}...")
    
    # Parse conversation history if provided
    history_context = ""
    if conversation_history:
        logger.info(f"Including conversation history ({len(conversation_history)} chars)")
        history_context = f"\n\nPrevious Conversation Context:\n{conversation_history}\n"
    
    progress_log = []
    
    logger.info("→ Step 1: Connecting to Qdrant")
    progress_log.append("→ Connecting to Qdrant...")
    client = get_qdrant_client(QDRANT_URL, QDRANT_API_KEY)
    logger.info("✓ Qdrant client connected")
    progress_log.append("✓ Connected to Qdrant")
    
    logger.info("→ Step 2: Generating query embedding")
    progress_log.append("→ Generating query embedding...")
    query_embedding = get_embedding(query, OPENAI_API_KEY)
    logger.info(f"✓ Query embedding generated (dimension: {len(query_embedding)})")
    progress_log.append(f"✓ Query embedding generated")
    
    logger.info(f"→ Step 3: Searching for top {top_k} relevant documents")
    progress_log.append(f"→ Searching for top {top_k} relevant documents...")
    search_results = search_documents_with_metadata(client, COLLECTION_NAME, query_embedding, top_k)
    
    if not search_results:
        logger.warning("✗ No documents found in collection")
        progress_log.append("✗ No documents found")
        return {
            "code": "No documents found. Please upload reference documents first.", 
            "error": "no_documents", 
            "progress": progress_log,
            "sources": []
        }
    
    logger.info(f"✓ Found {len(search_results)} relevant document chunks")
    progress_log.append(f"✓ Found {len(search_results)} relevant chunks")
    
    # Log search results with sources
    for i, result in enumerate(search_results, 1):
        page_info = f"page {result['page_number']}" if result['page_number'] else f"chunk {result['chunk_index']}"
        logger.info(f"  Source {i}: {result['source']} ({page_info}) - relevance: {result['score']:.4f}")
    
    logger.info("→ Step 4: Building context from retrieved documents")
    progress_log.append("→ Building context with citations...")
    
    # Build context with source markers
    context_parts = []
    for i, result in enumerate(search_results, 1):
        page_info = f"Page {result['page_number']}" if result['page_number'] else f"Chunk {result['chunk_index']}"
        context_parts.append(f"[Source {i}: {result['source']} - {page_info}]\n{result['text']}")
    
    context = "\n\n---\n\n".join(context_parts)
    logger.info(f"✓ Context built with {len(search_results)} sources ({len(context)} characters)")
    progress_log.append(f"✓ Context built with {len(search_results)} sources")
    
    logger.info("→ Step 5: Building strict grounded prompt")
    progress_log.append("→ Building strict grounded prompt...")
    
    # Create strict anti-hallucination prompt with conversation context
    strict_prompt = f"""You are a precise code generation assistant. You MUST follow these rules strictly:

1. ONLY use information from the provided reference documents
2. DO NOT add any information not present in the references
3. If the references don't contain enough information, say "The provided documentation does not contain sufficient information for..."
4. When generating code, cite which source number [Source N] you're using
5. Include comments in code indicating which source the logic comes from
6. If this is a follow-up question, consider the previous conversation context but still ground responses in the reference documents

{history_context}
Reference Documents:
{context}

User Query:
{query}

Instructions:
- Generate code based ONLY on the reference documents above
- Add comments like "# Based on Source 1: filename.pdf - Page X"
- If multiple approaches are mentioned in different sources, mention all of them
- If information is missing, explicitly state what's missing
- Include inline citations in comments
- For follow-up questions, maintain context from the previous conversation while staying grounded in the documents

Generate the code with citations:"""
    
    logger.info(f"✓ Strict prompt built ({len(strict_prompt)} characters)")
    progress_log.append("✓ Strict anti-hallucination prompt ready")
    
    logger.info("→ Step 6: Generating grounded code with GPT-4")
    progress_log.append("→ Generating grounded code with GPT-4...")
    
    # Generate with strict parameters to reduce hallucination
    code = generate_code(context, strict_prompt, OPENAI_API_KEY)
    logger.info(f"✓ Code generated ({len(code)} characters)")
    progress_log.append(f"✓ Code generated with citations!")
    
    # Format sources for response
    sources = []
    for i, result in enumerate(search_results, 1):
        page_info = f"Page {result['page_number']}" if result['page_number'] else f"Chunk {result['chunk_index']}"
        sources.append({
            "id": i,
            "file": result['source'],
            "location": page_info,
            "page_number": result['page_number'],
            "chunk_index": result['chunk_index'],
            "relevance_score": result['score'],
            "excerpt": result['text'][:300] + "..." if len(result['text']) > 300 else result['text']
        })
    
    logger.info(f"========== Code generation completed successfully ==========")
    return {
        "code": code,
        "sources": sources,
        "progress": progress_log,
        "query": query,
        "num_sources": len(sources)
    }

# Streaming endpoint for real-time code generation
@app.post("/generate-stream")
async def generate_stream(
    query: str = Form(...),
    top_k: int = Form(5),
    conversation_history: str = Form("")
):
    """
    Generate code with streaming response for real-time display.
    Returns Server-Sent Events (SSE) stream.
    """
    logger.info(f"========== Starting streaming code generation pipeline ==========")
    logger.info(f"Query: {query[:100]}...")
    
    async def event_generator():
        try:
            # Step 1: Connect to Qdrant
            yield f"data: {json.dumps({'type': 'status', 'message': 'Connecting to Qdrant...'})}\n\n"
            client = get_qdrant_client(QDRANT_URL, QDRANT_API_KEY)
            logger.info("✓ Qdrant client connected")
            
            # Step 2: Generate query embedding
            yield f"data: {json.dumps({'type': 'status', 'message': 'Generating query embedding...'})}\n\n"
            query_embedding = get_embedding(query, OPENAI_API_KEY)
            logger.info("✓ Query embedding generated")
            
            # Step 3: Search for relevant documents
            yield f"data: {json.dumps({'type': 'status', 'message': f'Searching for top {top_k} relevant documents...'})}\n\n"
            search_results = search_documents_with_metadata(client, COLLECTION_NAME, query_embedding, top_k=top_k)
            
            if not search_results:
                yield f"data: {json.dumps({'type': 'error', 'message': 'No documents found. Please upload reference documents first.'})}\n\n"
                return
            
            logger.info(f"✓ Found {len(search_results)} relevant document chunks")
            
            # Send sources to frontend first
            sources = []
            for i, result in enumerate(search_results, 1):
                page_info = f"Page {result['page_number']}" if result['page_number'] else f"Chunk {result['chunk_index']}"
                sources.append({
                    "id": i,
                    "file": result['source'],
                    "location": page_info,
                    "page_number": result['page_number'],
                    "chunk_index": result['chunk_index'],
                    "relevance_score": result['score'],
                    "excerpt": result['text'][:300] + "..." if len(result['text']) > 300 else result['text']
                })
            
            yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"
            
            # Step 4: Build context
            yield f"data: {json.dumps({'type': 'status', 'message': 'Building context from retrieved documents...'})}\n\n"
            context_parts = []
            for i, result in enumerate(search_results, 1):
                page_info = f"Page {result['page_number']}" if result['page_number'] else f"Chunk {result['chunk_index']}"
                context_parts.append(f"[Source {i}: {result['source']} - {page_info}]\n{result['text']}")
            
            context = "\n\n---\n\n".join(context_parts)
            logger.info(f"✓ Context built with {len(search_results)} sources")
            
            # Step 5: Build prompt
            history_context = ""
            if conversation_history.strip():
                history_context = f"Previous Conversation Context:\n{conversation_history}\n\n"
            
            strict_prompt = f"""You are a precise code generation assistant. You MUST follow these rules strictly:

1. ONLY use information from the provided reference documents
2. DO NOT add any information not present in the references
3. If the references don't contain enough information, say "The provided documentation does not contain sufficient information for..."
4. When generating code, cite which source number [Source N] you're using
5. Include comments in code indicating which source the logic comes from
6. If this is a follow-up question, consider the previous conversation context but still ground responses in the reference documents

{history_context}
Reference Documents:
{context}

User Query:
{query}

Instructions:
- Generate code based ONLY on the reference documents above
- Add comments like "# Based on Source 1: filename.pdf - Page X"
- If multiple approaches are mentioned in different sources, mention all of them
- If information is missing, explicitly state what's missing
- Include inline citations in comments
- For follow-up questions, maintain context from the previous conversation while staying grounded in the documents

Generate the code with citations:"""
            
            # Step 6: Stream code generation
            yield f"data: {json.dumps({'type': 'status', 'message': 'Generating code with GPT-4...'})}\n\n"
            
            # Stream the code chunks
            for code_chunk in generate_code_stream(context, strict_prompt, OPENAI_API_KEY):
                yield f"data: {json.dumps({'type': 'code', 'content': code_chunk})}\n\n"
            
            # Send completion signal
            yield f"data: {json.dumps({'type': 'done', 'message': 'Code generation completed!'})}\n\n"
            logger.info("✓ Streaming code generation completed")
            
        except Exception as e:
            logger.error(f"Error in streaming: {str(e)}")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )

# Agentic workflow endpoint
@app.post("/agentic")
async def agentic_workflow(
    query: str = Form(...),
    scenario: str = Form(...)
):
    """
    Accepts a user query and scenario (bug fix, migration, upgrade, feature).
    Builds context and prompt specific to the scenario for code generation.
    """
    logger.info(f"========== Starting agentic workflow ==========")
    logger.info(f"Scenario: {scenario}")
    logger.info(f"Query: {query[:100]}...")
    
    logger.info("→ Retrieving relevant context")
    client = get_qdrant_client(QDRANT_URL, QDRANT_API_KEY)
    query_embedding = get_embedding(query, OPENAI_API_KEY)
    top_docs = search_documents(client, COLLECTION_NAME, query_embedding)
    
    if not top_docs:
        logger.warning("✗ No documents found for agentic workflow")
        return {"code": "No documents found. Please upload reference documents first.", "error": "no_documents", "scenario": scenario}
    
    logger.info(f"✓ Retrieved {len(top_docs)} relevant chunks")
    
    context = "\n\n".join(top_docs)
    scenario_instructions = {
        "bug_fix": "Focus on identifying and fixing the bug described. Output only the corrected code and a brief comment explaining the fix.",
        "migration": "Provide code to migrate from the old system or API to the new one. Highlight changes and ensure compatibility.",
        "upgrade": "Generate code to upgrade dependencies, libraries, or frameworks as described. Ensure backward compatibility where possible.",
        "feature": "Implement the new feature as described in the user story. Include necessary code, comments, and usage examples."
    }
    scenario_text = scenario_instructions.get(scenario, "Follow best practices for the requested scenario.")
    logger.info(f"→ Building scenario-specific prompt for: {scenario}")
    prompt = f"Reference:\n{context}\n\nInstruction:\n{query}\n\nScenario:\n{scenario}\n\n{scenario_text}\n"
    
    logger.info("→ Generating code with agentic approach")
    code = generate_code(context, prompt, OPENAI_API_KEY)
    logger.info(f"✓ Agentic code generated ({len(code)} characters)")
    logger.info(f"========== Agentic workflow completed ==========")
    
    return {"code": code, "scenario": scenario}

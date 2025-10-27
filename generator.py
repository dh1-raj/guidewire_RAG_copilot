import openai
import logging

logger = logging.getLogger(__name__)

def generate_code(context, query, openai_api_key):
    """Generate code using OpenAI GPT-4 API (non-streaming)."""
    logger.info("Starting code generation with GPT-4")
    logger.debug(f"Context length: {len(context)} characters")
    logger.debug(f"Query: {query[:100]}...")
    
    openai.api_key = openai_api_key
    prompt = f"Reference:\n{context}\n\nInstruction:\n{query}\n\nGenerate code as per the reference and instruction."
    
    logger.info(f"Sending request to GPT-4 (prompt length: {len(prompt)} characters)")
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    
    code = response["choices"][0]["message"]["content"]
    logger.info(f"Received response from GPT-4 (length: {len(code)} characters)")
    logger.debug(f"Token usage - Prompt: {response['usage']['prompt_tokens']}, Completion: {response['usage']['completion_tokens']}, Total: {response['usage']['total_tokens']}")
    
    return code

def generate_code_stream(context, query, openai_api_key):
    """Generate code using OpenAI GPT-4 API with streaming."""
    logger.info("Starting streaming code generation with GPT-4")
    logger.debug(f"Context length: {len(context)} characters")
    logger.debug(f"Query: {query[:100]}...")
    
    openai.api_key = openai_api_key
    prompt = f"Reference:\n{context}\n\nInstruction:\n{query}\n\nGenerate code as per the reference and instruction."
    
    logger.info(f"Sending streaming request to GPT-4 (prompt length: {len(prompt)} characters)")
    
    # Create streaming response
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        stream=True
    )
    
    # Stream the response
    for chunk in response:
        if chunk["choices"][0].get("delta", {}).get("content"):
            content = chunk["choices"][0]["delta"]["content"]
            yield content
    
    logger.info("Streaming code generation completed")

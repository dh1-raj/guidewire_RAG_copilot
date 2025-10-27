import os
import logging
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Template for code generation prompt
CODE_PROMPT_TEMPLATE = """
Reference:
{context}

Instruction:
{query}

Output Requirements:
- Code should be clear, well-commented, and follow best practices.
- Include necessary imports and environment setup.
- If relevant, add usage examples.
- Output only code, no explanations unless requested.
"""

def build_prompt(context, query):
    """
    Build the prompt for code generation using retrieved context and user query.
    """
    logger.debug(f"Building prompt with context length: {len(context)}, query length: {len(query)}")
    prompt = CODE_PROMPT_TEMPLATE.format(context=context, query=query)
    logger.debug(f"Generated prompt length: {len(prompt)}")
    return prompt

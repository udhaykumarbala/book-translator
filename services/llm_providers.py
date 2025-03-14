import os
import httpx
import logging
import time
import openai
from openai import OpenAI
import anthropic
from typing import Dict, Any, List, Optional
import json
import redis

# Configure logging
logger = logging.getLogger(__name__)

# Set API keys from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

# Initialize Redis for caching if available
redis_url = os.getenv("REDIS_URL")
redis_client = None
if redis_url:
    try:
        redis_client = redis.from_url(redis_url)
        logger.info("Redis cache initialized successfully")
    except Exception as e:
        logger.warning(f"Failed to initialize Redis cache: {str(e)}")

# Initialize OpenAI client with explicit httpx client to avoid proxy issues
http_client = httpx.Client()
client = OpenAI(
    api_key=openai_api_key,
    http_client=http_client
)

# Language code mappings for different services
LANGUAGE_CODE_MAPPING = {
    'ta': {
        'google': 'ta',
        'internal': 'tamil'
    },
    'hi': {
        'google': 'hi',
        'internal': 'hindi'
    },
    'mr': {
        'google': 'mr',
        'internal': 'marathi'
    }
}

def get_openai_translation(system_prompt: str, user_prompt: str) -> str:
    """
    Get translation using OpenAI's API.
    
    Args:
        system_prompt: The system instructions for the translation task
        user_prompt: The text to translate
        
    Returns:
        The translated text
    """
    start_time = time.time()
    logger.info("Starting OpenAI translation request")
    logger.info(f"System prompt length: {len(system_prompt)}, User prompt length: {len(user_prompt)}")
    
    try:
        logger.info("Sending request to OpenAI API (model: gpt-4o)")
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=4000
        )
        
        result = response.choices[0].message.content.strip()
        logger.info(f"OpenAI response received in {time.time() - start_time:.2f} seconds")
        logger.info(f"Response length: {len(result)} characters")
        
        return result
    except Exception as e:
        logger.error(f"OpenAI API Error: {str(e)}", exc_info=True)
        raise Exception(f"OpenAI translation failed: {str(e)}")

def get_anthropic_translation(system_prompt: str, user_prompt: str) -> str:
    """
    Get translation using Anthropic's API.
    
    Args:
        system_prompt: The system instructions for the translation task
        user_prompt: The text to translate
        
    Returns:
        The translated text
    """
    start_time = time.time()
    logger.info("Starting Anthropic translation request")
    logger.info(f"System prompt length: {len(system_prompt)}, User prompt length: {len(user_prompt)}")
    
    try:
        logger.info("Initializing Anthropic client")
        client = anthropic.Anthropic(api_key=anthropic_api_key)
        
        prompt = f"{system_prompt}\n\n{user_prompt}"
        
        logger.info("Sending request to Anthropic API (model: claude-3-sonnet)")
        response = client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=4000,
            temperature=0.3,
            system="You are an expert literary translator focused on preserving the essence of stories.",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        result = response.content[0].text
        logger.info(f"Anthropic response received in {time.time() - start_time:.2f} seconds")
        logger.info(f"Response length: {len(result)} characters")
        
        return result
    except Exception as e:
        logger.error(f"Anthropic API Error: {str(e)}", exc_info=True)
        raise Exception(f"Anthropic translation failed: {str(e)}")

def perform_two_stage_translation(system_prompt_stage1: str, system_prompt_stage2: str, 
                                 input_text: str, provider: str = 'openai') -> str:
    """
    Performs a two-stage translation process to improve semantic preservation.
    Stage 1: Direct translation focused on semantic accuracy
    Stage 2: Refinement focused on natural language while preserving meaning
    
    Args:
        system_prompt_stage1: System prompt for the first stage
        system_prompt_stage2: System prompt for the second stage
        input_text: The text to translate
        provider: The LLM provider to use ('openai' or 'anthropic')
        
    Returns:
        The two-stage translated text
    """
    start_time = time.time()
    logger.info(f"Starting two-stage translation with provider: {provider}")
    
    try:
        # Stage 1: Initial translation with focus on semantic preservation
        logger.info("Starting stage 1 translation (semantic preservation)")
        stage1_start = time.time()
        
        if provider == 'openai':
            stage1_translation = get_openai_translation(system_prompt_stage1, input_text)
        elif provider == 'anthropic':
            stage1_translation = get_anthropic_translation(system_prompt_stage1, input_text)
        else:
            logger.error(f"Unsupported provider for two-stage translation: {provider}")
            raise ValueError(f"Unsupported provider for two-stage translation: {provider}")
        
        logger.info(f"Stage 1 translation completed in {time.time() - stage1_start:.2f} seconds")
        logger.info(f"Stage 1 result length: {len(stage1_translation)} characters")
            
        # Create combined prompt for stage 2 that includes both original and stage 1 translation
        stage2_user_prompt = (
            f"Original text:\n\n{input_text}\n\n"
            f"Initial translation:\n\n{stage1_translation}\n\n"
            f"Please refine this translation as instructed, making it more natural "
            f"while preserving ALL semantic meaning from the original text."
        )
        
        # Stage 2: Refinement with focus on naturalness while preserving meaning
        logger.info("Starting stage 2 translation (refinement)")
        stage2_start = time.time()
        
        if provider == 'openai':
            stage2_translation = get_openai_translation(system_prompt_stage2, stage2_user_prompt)
        elif provider == 'anthropic':
            stage2_translation = get_anthropic_translation(system_prompt_stage2, stage2_user_prompt)
        
        logger.info(f"Stage 2 translation completed in {time.time() - stage2_start:.2f} seconds")
        logger.info(f"Stage 2 result length: {len(stage2_translation)} characters")
        logger.info(f"Total two-stage translation time: {time.time() - start_time:.2f} seconds")
            
        return stage2_translation
    except Exception as e:
        logger.error(f"Two-stage translation Error: {str(e)}", exc_info=True)
        # If there's an error in the two-stage process, fall back to single-stage
        logger.warning("Falling back to single-stage translation")
        if provider == 'openai':
            return get_openai_translation(system_prompt_stage1, input_text)
        else:
            return get_anthropic_translation(system_prompt_stage1, input_text)

def perform_three_stage_translation(system_prompt_stage1: str, system_prompt_stage2: str, 
                                   system_prompt_stage3: str, input_text: str, 
                                   input_language: str, output_language: str,
                                   provider: str = 'openai') -> str:
    """
    Performs a three-stage translation process to create native-like text that preserves essence.
    Stage 1: Initial translation focused on semantic accuracy and complete preservation
    Stage 2: Refinement focused on cultural adaptation and natural language
    Stage 3: Final polish focused on making the text read like it was originally written in target language
    
    Args:
        system_prompt_stage1: System prompt for the first stage (semantic preservation)
        system_prompt_stage2: System prompt for the second stage (cultural adaptation)
        system_prompt_stage3: System prompt for the third stage (native writer polish)
        input_text: The text to translate
        input_language: The source language code
        output_language: The target language code
        provider: The LLM provider to use ('openai' or 'anthropic')
        
    Returns:
        The three-stage translated text with native-like quality
    """
    start_time = time.time()
    logger.info(f"Starting three-stage translation with provider: {provider}")
    
    try:
        # Stage 1: Initial translation with focus on semantic preservation
        logger.info("Starting stage 1 translation (semantic preservation)")
        stage1_start = time.time()
        
        if provider == 'openai':
            stage1_translation = get_openai_translation(system_prompt_stage1, input_text)
        elif provider == 'anthropic':
            stage1_translation = get_anthropic_translation(system_prompt_stage1, input_text)
        else:
            logger.error(f"Unsupported provider for three-stage translation: {provider}")
            raise ValueError(f"Unsupported provider for three-stage translation: {provider}")
        
        logger.info(f"Stage 1 translation completed in {time.time() - stage1_start:.2f} seconds")
        logger.info(f"Stage 1 result length: {len(stage1_translation)} characters")
            
        # Create combined prompt for stage 2 that includes both original and stage 1 translation
        stage2_user_prompt = (
            f"Original text ({input_language}):\n\n{input_text}\n\n"
            f"Initial translation ({output_language}):\n\n{stage1_translation}\n\n"
            f"Please refine this translation as instructed, adapting cultural elements "
            f"while preserving the semantic meaning."
        )
        
        # Stage 2: Refinement with focus on cultural adaptation
        logger.info("Starting stage 2 translation (cultural adaptation)")
        stage2_start = time.time()
        
        if provider == 'openai':
            stage2_translation = get_openai_translation(system_prompt_stage2, stage2_user_prompt)
        elif provider == 'anthropic':
            stage2_translation = get_anthropic_translation(system_prompt_stage2, stage2_user_prompt)
        
        logger.info(f"Stage 2 translation completed in {time.time() - stage2_start:.2f} seconds")
        logger.info(f"Stage 2 result length: {len(stage2_translation)} characters")
        
        # Create combined prompt for stage 3 that includes original, stage 1, and stage 2 translations
        stage3_user_prompt = (
            f"Original text ({input_language}):\n\n{input_text}\n\n"
            f"Semantic translation ({output_language}):\n\n{stage1_translation}\n\n"
            f"Culturally adapted translation ({output_language}):\n\n{stage2_translation}\n\n"
            f"Please apply the final polish as instructed, making this text read "
            f"like it was originally written by a native {output_language} author "
            f"while maintaining the essence of the original."
        )
        
        # Stage 3: Final polish with focus on native-like quality
        logger.info("Starting stage 3 translation (native writer polish)")
        stage3_start = time.time()
        
        if provider == 'openai':
            stage3_translation = get_openai_translation(system_prompt_stage3, stage3_user_prompt)
        elif provider == 'anthropic':
            stage3_translation = get_anthropic_translation(system_prompt_stage3, stage3_user_prompt)
        
        logger.info(f"Stage 3 translation completed in {time.time() - stage3_start:.2f} seconds")
        logger.info(f"Stage 3 result length: {len(stage3_translation)} characters")
        logger.info(f"Total three-stage translation time: {time.time() - start_time:.2f} seconds")
            
        return stage3_translation
    except Exception as e:
        logger.error(f"Three-stage translation Error: {str(e)}", exc_info=True)
        # If there's an error in the three-stage process, fall back to two-stage
        logger.warning("Falling back to two-stage translation")
        try:
            return perform_two_stage_translation(system_prompt_stage1, system_prompt_stage2, input_text, provider)
        except Exception as fallback_error:
            logger.error(f"Two-stage fallback error: {str(fallback_error)}", exc_info=True)
            # If two-stage fails, fall back to single-stage
            logger.warning("Falling back to single-stage translation")
            if provider == 'openai':
                return get_openai_translation(system_prompt_stage1, input_text)
            else:
                return get_anthropic_translation(system_prompt_stage1, input_text)

def use_local_llm_api(prompt: str) -> str:
    """
    Utilize the local LLM API for translations when external APIs are unavailable.
    This function connects to the local LLM API provided in the tools directory.
    
    Args:
        prompt: The full prompt including system and user instructions
        
    Returns:
        The translated text from the local LLM
    """
    start_time = time.time()
    logger.info("Starting local LLM API request")
    logger.info(f"Prompt length: {len(prompt)} characters")
    
    try:
        # Import the local LLM API module
        import sys
        import subprocess
        from pathlib import Path
        
        # Call the local LLM API
        logger.info("Executing local LLM API call with anthropic provider")
        result = subprocess.run(
            ["venv/bin/python", "./tools/llm_api.py", "--prompt", prompt, "--provider", "anthropic"],
            capture_output=True,
            text=True,
            check=True
        )
        
        # Extract the response
        output = result.stdout.strip()
        logger.info(f"Local LLM response received in {time.time() - start_time:.2f} seconds")
        logger.info(f"Response length: {len(output)} characters")
        
        return output
    except Exception as e:
        logger.error(f"Local LLM API Error: {str(e)}", exc_info=True)
        raise Exception(f"Local LLM translation failed: {str(e)}")

def get_cached_translation(text: str, lang: str, priority: int) -> Optional[str]:
    """
    Check for cached translation in Redis.
    
    Args:
        text: Source text to translate
        lang: Target language code
        priority: Priority value
        
    Returns:
        Cached translation if available, None otherwise
    """
    if not redis_client:
        return None
        
    cache_key = f"trans:{hash(text)}:{lang}:{priority}"
    try:
        cached_result = redis_client.get(cache_key)
        if cached_result:
            logger.info(f"Cache hit for {cache_key[:20]}...")
            return cached_result.decode('utf-8')
        logger.info(f"Cache miss for {cache_key[:20]}...")
        return None
    except Exception as e:
        logger.warning(f"Redis cache error: {str(e)}")
        return None
        
def cache_translation(text: str, lang: str, priority: int, result: str, ttl: int = 3600) -> None:
    """
    Cache translation result in Redis.
    
    Args:
        text: Source text that was translated
        lang: Target language code
        priority: Priority value
        result: Translation result to cache
        ttl: Time to live in seconds (default: 1 hour)
    """
    if not redis_client:
        return
        
    cache_key = f"trans:{hash(text)}:{lang}:{priority}"
    try:
        redis_client.setex(cache_key, ttl, result)
        logger.info(f"Cached translation for {cache_key[:20]}... (TTL: {ttl}s)")
    except Exception as e:
        logger.warning(f"Redis cache error: {str(e)}")

def generate_priority_prompt(text: str, target_lang: str, priority: int) -> str:
    """
    Generate an optimized prompt based on priority level.
    
    Args:
        text: Text to translate
        target_lang: Target language code
        priority: Priority value (0-100)
        
    Returns:
        Optimized prompt for translation
    """
    base_instruction = f"Translate this literary text from English to {target_lang} as if written by a native author, preserving: "
    
    priority_level = 'high' if priority > 70 else 'low' if priority < 30 else 'mid'
    
    priority_map = {
        'high': {
            'instruction': "Prioritize cultural authenticity and natural flow over literal accuracy. ",
            'style': "Use colloquial expressions and adapt idioms to local equivalents.",
            'examples': f"Example: 'He kicked the bucket' → use regional equivalent of 'passed away' in {target_lang}"
        },
        'mid': {
            'instruction': "Balance semantic accuracy with readable prose. ",
            'style': "Maintain original structure while using natural phrasing.",
            'examples': ""
        },
        'low': {
            'instruction': "Prioritize exact meaning preservation. ",
            'style': "Maintain English sentence structure when necessary.",
            'examples': ""
        }
    }
    
    components = priority_map[priority_level]
    
    return f"""
    {base_instruction}
    {components['instruction']}
    - Translate cultural references to {target_lang} equivalents
    - Use {target_lang}-specific idioms and proverbs
    - Maintain author's original tone (formal/informal)
    {components['style']}
    {components['examples']}
    
    Text: {text}
    """

def enhanced_translation_pipeline(text: str, source_lang: str, target_lang: str, priority: int) -> str:
    """
    Multi-stage LLM orchestration translation pipeline based on priority.
    
    Args:
        text: Text to translate
        source_lang: Source language code
        target_lang: Target language code
        priority: Priority value (0-100)
        
    Returns:
        Translated text
    """
    logger.info(f"Starting enhanced translation pipeline for {source_lang} to {target_lang} with priority {priority}")
    
    # Check cache first
    cached_result = get_cached_translation(text, target_lang, priority)
    if cached_result:
        logger.info("Returning cached translation")
        return cached_result
    
    # Route based on priority
    if priority >= 70:  # High priority - focus on native quality
        logger.info("Using high-priority native-focused translation pipeline")
        
        # Stage 1: Semantic Preservation with GPT-4o
        stage1_prompt = f"Literal translation from {source_lang} to {target_lang} preserving exact meaning. Maintain ALL details and nuances."
        stage1_result = get_openai_translation(stage1_prompt, text)
        
        # Stage 2: Cultural Adaptation with Claude 3
        stage2_prompt = f"Adapt this {target_lang} text using cultural equivalents while maintaining overall meaning. Make it feel native to {target_lang} culture."
        final_result = get_anthropic_translation(stage2_prompt, stage1_result)
            
    elif priority >= 30:  # Medium priority - balanced approach
        logger.info("Using medium-priority balanced translation pipeline")
        
        # Use the optimized prompt with Claude 3
        balanced_prompt = generate_priority_prompt(text, target_lang, priority)
        final_result = get_anthropic_translation(
            "You are a literary translator focused on balancing cultural adaptation with meaning preservation.",
            balanced_prompt
        )
        
    else:  # Low priority - focus on essence preservation
        logger.info("Using low-priority essence-preserving translation pipeline")
        
        # Use GPT-4o with specific prompt for essence preservation
        essence_prompt = generate_priority_prompt(text, target_lang, priority)
        final_result = get_openai_translation(
            "You are a precision translator ensuring maximum semantic preservation.",
            essence_prompt
        )
    
    # Cache the result
    cache_translation(text, target_lang, priority, final_result)
    
    return final_result 
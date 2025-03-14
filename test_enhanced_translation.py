import os
import pytest
import json
from unittest.mock import patch, Mock
from services.llm_providers import (
    generate_priority_prompt, 
    enhanced_translation_pipeline,
    get_cached_translation,
    cache_translation
)

# Sample test data
TEST_TEXT = "The old man and the sea is a story about a fisherman who struggles with a giant marlin."
TEST_SOURCE_LANG = "en"
TEST_TARGET_LANG = "ta"

@pytest.fixture
def mock_environment():
    """Set up test environment variables."""
    os.environ["OPENAI_API_KEY"] = "test-key"
    os.environ["ANTHROPIC_API_KEY"] = "test-key"
    yield
    # Clean up
    del os.environ["OPENAI_API_KEY"]
    del os.environ["ANTHROPIC_API_KEY"]

def test_generate_priority_prompt():
    """Test priority-specific prompt generation."""
    # Test high priority prompt
    high_prompt = generate_priority_prompt(TEST_TEXT, TEST_TARGET_LANG, 75)
    assert "Prioritize cultural authenticity" in high_prompt
    assert "native author" in high_prompt
    
    # Test medium priority prompt
    mid_prompt = generate_priority_prompt(TEST_TEXT, TEST_TARGET_LANG, 50)
    assert "Balance semantic accuracy" in mid_prompt
    
    # Test low priority prompt
    low_prompt = generate_priority_prompt(TEST_TEXT, TEST_TARGET_LANG, 25)
    assert "Prioritize exact meaning preservation" in low_prompt

@patch('services.llm_providers.get_cached_translation')
@patch('services.llm_providers.get_openai_translation')
@patch('services.llm_providers.cache_translation')
def test_enhanced_translation_pipeline_low_priority(
    mock_cache, mock_openai, mock_get_cached, mock_environment
):
    """Test the enhanced translation pipeline with low priority."""
    # Configure mocks
    mock_get_cached.return_value = None
    mock_openai.return_value = "Translated text"
    
    # Test with low priority (focus on essence preservation)
    result = enhanced_translation_pipeline(
        text=TEST_TEXT,
        source_lang=TEST_SOURCE_LANG,
        target_lang=TEST_TARGET_LANG,
        priority=20
    )
    
    # Verify OpenAI was called with essence-preserving prompt
    assert mock_openai.called
    assert mock_cache.called
    assert result == "Translated text"
    
    # Check that the first argument to OpenAI contains essence preservation
    system_prompt = mock_openai.call_args[0][0]
    assert "precision translator" in system_prompt

@patch('services.llm_providers.get_cached_translation')
@patch('services.llm_providers.get_anthropic_translation')
@patch('services.llm_providers.cache_translation')
def test_enhanced_translation_pipeline_medium_priority(
    mock_cache, mock_anthropic, mock_get_cached, mock_environment
):
    """Test the enhanced translation pipeline with medium priority."""
    # Configure mocks
    mock_get_cached.return_value = None
    mock_anthropic.return_value = "Translated text with balance"
    
    # Test with medium priority (balanced approach)
    result = enhanced_translation_pipeline(
        text=TEST_TEXT,
        source_lang=TEST_SOURCE_LANG,
        target_lang=TEST_TARGET_LANG,
        priority=50
    )
    
    # Verify Anthropic was called with balanced prompt
    assert mock_anthropic.called
    assert mock_cache.called
    assert result == "Translated text with balance"
    
    # Check that the first argument to Anthropic contains balanced approach
    system_prompt = mock_anthropic.call_args[0][0]
    assert "balancing cultural adaptation" in system_prompt

@patch('services.llm_providers.get_cached_translation')
@patch('services.llm_providers.get_openai_translation')
@patch('services.llm_providers.get_anthropic_translation')
@patch('services.llm_providers.cache_translation')
def test_enhanced_translation_pipeline_high_priority(
    mock_cache, mock_anthropic, mock_openai, mock_get_cached, mock_environment
):
    """Test the enhanced translation pipeline with high priority."""
    # Configure mocks
    mock_get_cached.return_value = None
    mock_openai.return_value = "Stage 1 translation"
    mock_anthropic.return_value = "Stage 2 translation"
    
    # Test with high priority (focus on native quality)
    result = enhanced_translation_pipeline(
        text=TEST_TEXT,
        source_lang=TEST_SOURCE_LANG,
        target_lang=TEST_TARGET_LANG,
        priority=80
    )
    
    # Verify the 2-stage pipeline was used
    assert mock_openai.called
    assert mock_anthropic.called
    assert mock_cache.called
    assert result == "Stage 2 translation"

@patch('services.llm_providers.redis_client')
def test_translation_caching(mock_redis, mock_environment):
    """Test the translation caching mechanism."""
    # Configure mock
    mock_redis.get.return_value = b"Cached translation result"
    mock_redis.setex.return_value = True
    
    # Test cache retrieval
    cached_result = get_cached_translation(
        text=TEST_TEXT,
        lang=TEST_TARGET_LANG,
        priority=50
    )
    
    assert cached_result == "Cached translation result"
    assert mock_redis.get.called
    
    # Test cache storage
    cache_translation(
        text=TEST_TEXT,
        lang=TEST_TARGET_LANG,
        priority=50,
        result="New translation result"
    )
    
    assert mock_redis.setex.called

if __name__ == '__main__':
    pytest.main(["-xvs", __file__]) 
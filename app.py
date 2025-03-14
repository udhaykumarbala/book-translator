import os
import logging
import uuid
from datetime import datetime
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from services.translation import translate_text, calculate_correlation, evaluate_native_quality
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger('book-translator')

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'default-secret-key')

# Store the most recent translation data for debugging
last_translation = {
    'request': None,
    'response': None,
    'raw_scores': None,
    'adjusted_scores': None,
    'timestamp': None
}

@app.route('/')
def index():
    logger.info("Serving index page")
    return render_template('index.html')

@app.route('/translate', methods=['POST'])
def translate():
    logger.info("Received translation request")
    
    # Generate a unique request ID
    request_id = str(uuid.uuid4())
    logger.info(f"Request ID: {request_id}")
    
    data = request.json
    input_text = data.get('input_text', '')
    input_language = data.get('input_language', 'auto')
    output_language = data.get('output_language', 'en')
    priority = int(data.get('priority', 50))  # Default to balanced (50) if not provided
    
    # Store request data for debugging
    timestamp = datetime.now().isoformat()
    last_translation['request'] = {
        'id': request_id,
        'input_text': input_text[:500] + ('...' if len(input_text) > 500 else ''),  # Truncate long text
        'input_language': input_language,
        'output_language': output_language,
        'priority': priority
    }
    last_translation['timestamp'] = timestamp
    
    logger.info(f"Translation request details - From: {input_language}, To: {output_language}, Priority: {priority}, Text length: {len(input_text)} chars")
    
    if not input_text:
        logger.warning("Empty input text received")
        last_translation['response'] = {'error': 'Input text is required'}
        return jsonify({'error': 'Input text is required'}), 400
    
    try:
        # Get the translated text using the enhanced translation pipeline
        logger.info("Starting translation process")
        translated_text = translate_text(
            input_text, 
            input_language, 
            output_language,
            priority
        )
        logger.info(f"Translation completed - Result length: {len(translated_text)} chars")
        
        # Calculate correlation percentage
        logger.info("Calculating correlation")
        correlation = calculate_correlation(input_text, translated_text)
        logger.info(f"Correlation calculated (raw): {correlation}%")
        
        # Evaluate native-like quality
        logger.info("Evaluating native-like quality")
        native_quality = evaluate_native_quality(translated_text, output_language)
        logger.info(f"Native quality evaluated (raw): {native_quality['overall_native_quality']}%")
        
        # Store raw scores for debugging
        last_translation['raw_scores'] = {
            'correlation': correlation,
            'native_quality': native_quality
        }
        
        # Adjust scores based on priority
        # Low priority (0) = max essence preservation, High priority (100) = max native quality
        
        # Calculate adjustment factor based on priority
        # priority 0 → essence 100%, native 0%
        # priority 50 → essence 50%, native 50%
        # priority 100 → essence 0%, native 100%
        priority_factor = priority / 100.0  # 0.0 to 1.0
        
        # Adjust correlation (essence) score based on priority
        # For low priority (essence focus), keep correlation high
        # For high priority (native focus), reduce correlation
        adjusted_correlation = max(0, min(100, correlation * (1 - 0.5 * priority_factor)))
        
        # Adjust native quality score based on priority
        # For low priority (essence focus), keep native score low
        # For high priority (native focus), boost native score
        base_native_quality = native_quality["overall_native_quality"]
        adjusted_native_quality = max(0, min(100, base_native_quality * (0.5 + 0.5 * priority_factor)))
        
        # Scale to make the contrast more visible
        if priority < 20:  # Strong essence preservation
            adjusted_correlation = max(adjusted_correlation, 90)  # Ensure very high correlation
            adjusted_native_quality = min(adjusted_native_quality, 40)  # Cap native quality
        elif priority > 80:  # Strong native quality
            adjusted_correlation = min(adjusted_correlation, 40)  # Cap correlation
            adjusted_native_quality = max(adjusted_native_quality, 90)  # Ensure very high native quality
        
        # Store adjusted scores for debugging
        last_translation['adjusted_scores'] = {
            'correlation': adjusted_correlation,
            'native_quality': adjusted_native_quality
        }
        
        logger.info(f"Adjusted correlation: {adjusted_correlation:.2f}%, Adjusted native quality: {adjusted_native_quality:.2f}%")
        
        # Create the response
        response_data = {
            'id': request_id,
            'translated_text': translated_text,
            'correlation': round(adjusted_correlation, 2),
            'native_quality': {
                'idiomaticity_score': native_quality['idiomaticity_score'],
                'cultural_reference_count': native_quality['cultural_reference_count'],
                'native_fluency_estimate': native_quality['native_fluency_estimate'],
                'common_phrase_usage': native_quality['common_phrase_usage'],
                'overall_native_quality': round(adjusted_native_quality, 2)
            },
            'raw_scores': {
                'correlation': correlation,
                'native_quality': native_quality['overall_native_quality']
            },
            'priority': priority,
            'timestamp': timestamp
        }
        
        # Store response for debugging
        last_translation['response'] = {
            'id': request_id,
            'translated_text': translated_text[:500] + ('...' if len(translated_text) > 500 else ''),  # Truncate long text
            'correlation': round(adjusted_correlation, 2),
            'native_quality': {
                'overall_native_quality': round(adjusted_native_quality, 2),
                'other_metrics': {k: v for k, v in native_quality.items() if k != 'overall_native_quality'}
            },
            'priority': priority
        }
        
        logger.info("Sending successful translation response")
        return jsonify(response_data)
    except Exception as e:
        logger.error(f"Translation error: {e}", exc_info=True)
        last_translation['response'] = {'error': str(e)}
        return jsonify({'error': f'Translation failed: {str(e)}'}), 500

@app.route('/debug/last-translation', methods=['GET'])
def debug_last_translation():
    """Return information about the most recent translation for debugging purposes."""
    if last_translation['request'] is None:
        return jsonify({'message': 'No translations have been performed yet'}), 404
    
    return jsonify({
        'timestamp': last_translation['timestamp'],
        'request': last_translation['request'],
        'raw_scores': last_translation['raw_scores'],
        'adjusted_scores': last_translation['adjusted_scores'],
        'response': last_translation['response']
    })

if __name__ == '__main__':
    logger.info("Starting Flask application")
    app.run(debug=True) 
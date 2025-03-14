# Book Translator - Essence Preserver

A Flask-based web application that translates text while preserving the essence, style, and cultural context of stories. Unlike traditional translation services that focus on word-for-word translation, this service is designed to convey the same emotions, cultural references, and stylistic elements in a way that resonates with readers of the target language.

## Features

- **Essence Preservation**: Maintains the author's voice, tone, and literary devices
- **Multiple Languages**: Supports translation between various languages
- **Correlation Percentage**: Provides a metric showing how well the essence is preserved
- **Modern UI**: Clean, responsive interface for easy interaction
- **Priority-Based Translation**: Adjust translation focus between essence preservation and native-like output
- **Multi-LLM Orchestration**: Uses different LLMs in orchestration for optimal results at different priority levels
- **Translation Memory**: Caches translations for improved performance and consistency
- **Enhanced Tamil Translation**: Specialized approach for Tamil language with higher essence preservation

## Technology Stack

- **Backend**: Python, Flask
- **Frontend**: HTML, CSS, JavaScript, Bootstrap 5
- **NLP**: Sentence Transformers, OpenAI GPT-4, Anthropic Claude
- **Metrics**: Cosine similarity for semantic correlation, native quality validation
- **Storage**: Redis for translation memory caching

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/book-translator.git
   cd book-translator
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the project root and add your API keys:
   ```
   FLASK_APP=app
   FLASK_ENV=development
   SECRET_KEY=your-secret-key
   OPENAI_API_KEY=your-openai-api-key
   ANTHROPIC_API_KEY=your-anthropic-api-key
   LLM_PROVIDER=openai  # Options: openai, anthropic
   REDIS_URL=redis://localhost:6379/0
   ```

## Usage

1. Start the Flask application:
   ```
   flask run
   ```

2. Open your browser and navigate to:
   ```
   http://localhost:5000
   ```

3. Enter the text you want to translate, select source and target languages, and click "Translate"

## How It Works

1. **Input Processing**: The application takes the input text and identifies its language (if not specified)
2. **Priority Routing**: Based on the priority setting (0-100), the system routes to the appropriate translation pipeline:
   - **Low Priority (0-30)**: Uses GPT-4o with specialized prompts for maximum semantic preservation
   - **Medium Priority (30-70)**: Uses Claude 3 with balanced prompts for readability with essence preservation
   - **High Priority (70-100)**: Uses a 2-stage pipeline (GPT-4o→Claude 3) for native-like quality
3. **Translation Memory**: Checks if a similar text has been translated before with the same settings
4. **Native Quality Evaluation**: Uses LLM to evaluate how native-like the translation appears
5. **Correlation Analysis**: Sentence embeddings calculate semantic similarity between original and translated text
6. **Result Display**: Displays the translated text with correlation and native quality metrics

## API Endpoints

- **GET /**: Main page
- **GET /languages**: Returns a list of supported languages
- **POST /translate**: Translates text with the following JSON parameters:
  - `input_text`: Text to translate
  - `input_language`: Source language code (or "auto" for auto-detection)
  - `output_language`: Target language code
  - `priority`: Value from 0-100 where 0 = max essence preservation and 100 = max native quality

## Running Tests

Run the test suite to verify the enhanced translation capabilities:

```
pytest test_enhanced_translation.py -v
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 
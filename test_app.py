import unittest
from app import app
import json
from unittest.mock import patch


class BookTranslatorTestCase(unittest.TestCase):
    def setUp(self):
        app.config['TESTING'] = True
        self.client = app.test_client()

    def test_index_route(self):
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)

    def test_languages_route(self):
        response = self.client.get('/languages')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIsInstance(data, list)
        self.assertTrue(len(data) > 0)
        self.assertIn('code', data[0])
        self.assertIn('name', data[0])

    @patch('services.translation.translate_text')
    @patch('services.translation.calculate_correlation')
    def test_translate_route(self, mock_correlation, mock_translate):
        # Mock translation and correlation
        mock_translate.return_value = "Translated text example"
        mock_correlation.return_value = 85.7
        
        # Test request
        request_data = {
            'input_text': 'This is a test text',
            'input_language': 'en',
            'output_language': 'fr'
        }
        
        response = self.client.post(
            '/translate',
            data=json.dumps(request_data),
            content_type='application/json'
        )
        
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('translated_text', data)
        self.assertIn('correlation', data)
        self.assertEqual(data['translated_text'], "Translated text example")
        self.assertEqual(data['correlation'], 85.7)

    def test_translate_route_missing_text(self):
        # Test request with missing text
        request_data = {
            'input_language': 'en',
            'output_language': 'fr'
        }
        
        response = self.client.post(
            '/translate',
            data=json.dumps(request_data),
            content_type='application/json'
        )
        
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertIn('error', data)


if __name__ == '__main__':
    unittest.main() 
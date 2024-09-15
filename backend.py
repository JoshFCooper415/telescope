from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
from io import BytesIO
from PIL import Image
import logging
import traceback
import torch
from aeroblade import AEROBLADE
from binoculars import Binoculars

# Configuration
BINOCULARS_MODEL_PERFORMER_NAME = "HuggingFaceTB/SmolLM-360M-Instruct"
BINOCULARS_MODEL_OBSERVER_NAME = "HuggingFaceTB/SmolLM-360M"

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize detectors
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
with open("hugging_face_auth_token.txt") as file:
    hugging_face_auth_token = file.readline().strip()

aeroblade_detector = AEROBLADE()
text_detector = Binoculars(BINOCULARS_MODEL_OBSERVER_NAME, BINOCULARS_MODEL_PERFORMER_NAME, hugging_face_auth_token)

@app.route('/analyze', methods=['POST', 'OPTIONS'])
def analyze():
    if request.method == 'OPTIONS':
        return '', 204
   
    try:
        data = request.json
        if not data:
            logger.error("No JSON data received")
            return jsonify({'error': 'No JSON data received'}), 400
       
        content_type = data.get('type')
        content = data.get('content')
       
        if content_type == "image":
            return analyze_image(content)
        elif content_type == "text":
            return analyze_text(content)
        else:
            logger.error(f"Invalid content type: {content_type}")
            return jsonify({'error': 'Invalid content type'}), 400
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': f'Unexpected error: {str(e)}'}), 500

def analyze_image(image_data):
    try:
        # Handle both raw base64 and data URL formats
        if ',' in image_data:
            image_data = image_data.split(',', 1)[1]
        image_data = base64.b64decode(image_data)
        image = Image.open(BytesIO(image_data)).convert('RGB')
    except Exception as e:
        logger.error(f"Failed to process image: {str(e)}")
        return jsonify({'error': f'Failed to process image: {str(e)}'}), 400
    try:
        detection_result, error = aeroblade_detector.detect_aeroblade(image)
        result = {
            'aeroblade': {
                'reconstruction_error': float(error),
                'result': f"{detection_result} (Telescope Score: {error:.4f})"
            }
        }
        return jsonify(result)
    except Exception as e:
        logger.error(f"Failed to analyze image with AEROBLADE: {str(e)}")
        return jsonify({'error': f'Failed to analyze image with AEROBLADE: {str(e)}'}), 500

def analyze_text(text_data):
    try:
        is_ai_generated, telescope_score = text_detector.predict(text_data, device)
        result = {
            'aeroblade': {
                'is_ai_generated': is_ai_generated,
                'telescope_score': float(telescope_score),
                'result': f"{'AI-generated' if is_ai_generated else 'Not AI-generated'} (Telescope Score: {float(telescope_score):.4f})"
            }
        }
        return jsonify(result)
    except Exception as e:
        logger.error(f"Failed to analyze text: {str(e)}")
        return jsonify({'error': f'Failed to analyze text: {str(e)}'}), 500

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5080, debug=True)
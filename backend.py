from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
from io import BytesIO
from PIL import Image
import torch
import torchvision.transforms as transforms
from lpips import LPIPS
from diffusers import AutoencoderKL
import logging
import traceback

from aeroblade import AEROBLADE
from binoculars import Binoculars

app = Flask(__name__)
CORS(app)

BINOCULARS_MODEL_PERFORMER_NAME = "HuggingFaceTB/SmolLM-360M-Instruct"
BINOCULARS_MODEL_OBSERVER_NAME = "HuggingFaceTB/SmolLM-360M"

with open("hugging_face_auth_token.txt") as file:
    hugging_face_auth_token = file.readline()
    
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

aeroblade_detector = AEROBLADE()
text_detector = Binoculars(BINOCULARS_MODEL_OBSERVER_NAME, BINOCULARS_MODEL_PERFORMER_NAME, hugging_face_auth_token)


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

@app.route('/analyze', methods=['POST'])
def analyze():
    logger.info("Received /analyze request")
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
            return jsonify({'error': f'Received content type that is not either an image or text'}), 500

    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': f'Unexpected error: {str(e)}'}), 500


def analyze_image(image_data):
    try:
        image_data = base64.b64decode(image_data)
        logger.info("Successfully decoded base64 image")
    except Exception as e:
        logger.error(f"Failed to decode base64 image: {str(e)}")
        return jsonify({'error': f'Failed to decode base64 image: {str(e)}'}), 400

    try:
        image = Image.open(BytesIO(image_data)).convert('RGB')
        logger.info("Successfully opened image")
    except Exception as e:
        logger.error(f"Failed to open image: {str(e)}")
        return jsonify({'error': f'Failed to open image: {str(e)}'}), 400

    result = {}

    # AEROBLADE analysis
    try:
        detection_result, error = aeroblade_detector.detect_aeroblade(image)
        result['aeroblade'] = {
            'reconstruction_error': float(error),
            'result': f"{detection_result} (Telescope Score: {error:.4f})"
            }
    except Exception as e:
        logger.error(f"Failed to analyze image with AEROBLADE: {str(e)}")
        result['aeroblade'] = {'error': f'Failed to analyze image with AEROBLADE: {str(e)}'}

        logger.info(jsonify(result))
        return jsonify(result)

    logger.info(f"Analysis result: {result}")
    return jsonify(result)


def analyze_text(text_data):
    how_ai_generated_string, telescope_score = text_detector.predict(text_data, device)
    
    result = {}
    result['aeroblade'] = {
        'is_ai_generated': how_ai_generated_string,
        'telescope_score ': float(telescope_score),
        'result': f"{how_ai_generated_string} (Telescope Score : {float(telescope_score):.4f})"
    }
    
    return jsonify(result)


    
    
if __name__ == '__main__':
    app.run(debug=True)
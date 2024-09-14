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

app = Flask(__name__)
CORS(app)

with open("hugging_face_auth_token.txt") as file:
    hugging_face_auth_token = file.readline()

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class AEROBLADE:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing AEROBLADE...")
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Using device: {self.device}")
        
        try:
            self.lpips = LPIPS(net='vgg', version='0.1').to(self.device)
            self.lpips.eval()
            for param in self.lpips.parameters():
                param.requires_grad = False
            self.logger.info("LPIPS model loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load LPIPS model: {str(e)}")
            raise
        
        try:
            # Load multiple AEs as described in the paper
            self.aes = {
                'SD2': AutoencoderKL.from_pretrained("stabilityai/stable-diffusion-2-1", subfolder="vae", token=hugging_face_auth_token).to(self.device),
                'flx': AutoencoderKL.from_pretrained("black-forest-labs/FLUX.1-dev", subfolder="vae", token=hugging_face_auth_token).to(self.device)
            }
            self.logger.info("AE models loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load AE models: {str(e)}")
            raise
        
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
        ])

    @torch.no_grad()
    def compute_reconstruction_error(self, image):
        self.logger.info("Computing reconstruction error...")
        try:
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            errors = []
            for ae_name, ae in self.aes.items():
                latents = ae.encode(image_tensor).latent_dist.sample()
                reconstructed = ae.decode(latents).sample
                
                # Compute LPIPS distance
                error = self.lpips(image_tensor, reconstructed).item()
                errors.append(error)
                self.logger.debug(f"Error for AE {ae_name}: {error}")
            
            min_error = min(errors)
            self.logger.info(f"Minimum reconstruction error: {min_error}")
            return min_error
        except Exception as e:
            self.logger.error(f"Error in compute_reconstruction_error: {str(e)}")
            raise

    def detect_aeroblade(self, image, threshold_probably=0.01, threshold_possibly=0.03,threshold_definitely=0.008):
        self.logger.info("Detecting if image is AI-generated using AEROBLADE...")
        try:
            error = self.compute_reconstruction_error(image)
            if error < threshold_definitely:
                result = "Definitely AI-generated"
            if error < threshold_probably:
                result = "Probably AI-generated"
            elif error < threshold_possibly:
                result = "Probably not AI-generated"
            else:
                result = "Not AI-generated"
            
            self.logger.info(f"AEROBLADE detection result: {result}, error={error}")
            return result, error
        except Exception as e:
            self.logger.error(f"Error in AEROBLADE detect: {str(e)}")
            raise

detector = AEROBLADE()

@app.route('/analyze', methods=['POST'])
def analyze_image():
    logger.info("Received /analyze request")
    try:
        data = request.json
        if not data:
            logger.error("No JSON data received")
            return jsonify({'error': 'No JSON data received'}), 400

        image_data = data.get('content')
        content_type = data.get('type')

        if not image_data:
            logger.error("No image content provided")
            return jsonify({'error': 'No image content provided'}), 400
        if content_type != 'image':
            logger.error(f"Invalid content type: {content_type}")
            return jsonify({'error': 'Invalid content type. Expected image.'}), 400

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
            detection_result, error = detector.detect_aeroblade(image)
            result['aeroblade'] = {
                'result': detection_result,
                'reconstruction_error': float(error),
                'details': f"{detection_result} (Error: {error:.4f})"
            }
        except Exception as e:
            logger.error(f"Failed to analyze image with AEROBLADE: {str(e)}")
            result['aeroblade'] = {'error': f'Failed to analyze image with AEROBLADE: {str(e)}'}

        logger.info(f"Analysis result: {result}")
        return jsonify(result)

    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': f'Unexpected error: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
from io import BytesIO
from PIL import Image
import torch
import torchvision.transforms as transforms
from lpips import LPIPS
from diffusers import AutoencoderKL
from transformers import AutoImageProcessor, AutoModelForImageClassification

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class AEROBLADE:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing AEROBLADE...")
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Using device: {self.device}")
        
        try:
            self.lpips = LPIPS(net='vgg').to(self.device)
            self.lpips.eval()
            for param in self.lpips.parameters():
                param.requires_grad = False
            self.logger.info("LPIPS model loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load LPIPS model: {str(e)}")
            raise
        
        try:
            self.vae_v2 = AutoencoderKL.from_pretrained("stabilityai/stable-diffusion-2-1", subfolder="vae").to(self.device)
            self.vae_v3 = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(self.device)
            self.logger.info("VAE models loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load VAE models: {str(e)}")
            raise
        
        try:
            self.hf_processor = AutoImageProcessor.from_pretrained("dima806/ai_vs_real_image_detection")
            self.hf_model = AutoModelForImageClassification.from_pretrained("dima806/ai_vs_real_image_detection").to(self.device)
            self.logger.info("HuggingFace model loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load HuggingFace model: {str(e)}")
            raise
        
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    @torch.no_grad()
    def compute_reconstruction_error(self, image):
        self.logger.info("Computing reconstruction error...")
        try:
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            errors = []
            for i, vae in enumerate([self.vae_v2, self.vae_v3]):
                latents = vae.encode(image_tensor).latent_dist.sample()
                reconstructed = vae.decode(latents).sample
                
                reconstructed = (reconstructed / 2 + 0.5).clamp(0, 1)
                
                error = self.lpips(image_tensor, reconstructed).item()
                errors.append(error)
                self.logger.debug(f"Error for VAE {i+1}: {error}")
            
            min_error = min(errors)
            self.logger.info(f"Minimum reconstruction error: {min_error}")
            return min_error
        except Exception as e:
            self.logger.error(f"Error in compute_reconstruction_error: {str(e)}")
            raise

    def detect_aeroblade(self, image, threshold=0.044):
        self.logger.info("Detecting if image is AI-generated using AEROBLADE...")
        try:
            error = self.compute_reconstruction_error(image)
            is_generated = error < threshold
            self.logger.info(f"AEROBLADE detection result: is_generated={is_generated}, error={error}")
            return is_generated, error
        except Exception as e:
            self.logger.error(f"Error in AEROBLADE detect: {str(e)}")
            raise

    @torch.no_grad()
    def detect_huggingface(self, image):
        self.logger.info("Detecting if image is AI-generated using HuggingFace model...")
        try:
            inputs = self.hf_processor(images=image, return_tensors="pt").to(self.device)
            outputs = self.hf_model(**inputs)
            probabilities = outputs.logits.softmax(dim=-1)[0]
            
            # Check if 'AI Generated' is in the label2id dictionary
            if 'AI Generated' in self.hf_model.config.label2id:
                ai_generated_index = self.hf_model.config.label2id['AI Generated']
            else:
                # If 'AI Generated' is not found, assume it's the second class (index 1)
                ai_generated_index = 1
            
            ai_generated_prob = probabilities[ai_generated_index].item()
            is_generated = ai_generated_prob > 0.5
            self.logger.info(f"HuggingFace detection result: is_generated={is_generated}, probability={ai_generated_prob}")
            return is_generated, ai_generated_prob
        except Exception as e:
            self.logger.error(f"Error in HuggingFace detect: {str(e)}")
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
        method = data.get('method', 'both')  # Default to both methods if not specified

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
        final_result = {}

        if method in ['huggingface', 'both']:
            try:
                is_generated_hf, probability = detector.detect_huggingface(image)
                result['huggingface'] = {
                    'is_ai_generated': is_generated_hf,
                    'ai_generated_probability': float(probability),
                    'result': f"{'AI-generated' if is_generated_hf else 'Not AI-generated'} (Probability: {probability:.4f})"
                }
                # Set the final result based on HuggingFace
                final_result = {
                    'is_ai_generated': is_generated_hf,
                    'confidence': float(probability),
                    'method': 'HuggingFace'
                }
            except Exception as e:
                logger.error(f"Failed to analyze image with HuggingFace model: {str(e)}")
                result['huggingface'] = {'error': f'Failed to analyze image with HuggingFace model: {str(e)}'}

        if method in ['aeroblade', 'both']:
            try:
                is_generated_aeroblade, error = detector.detect_aeroblade(image)
                result['aeroblade'] = {
                    'is_ai_generated': is_generated_aeroblade,
                    'reconstruction_error': float(error),
                    'result': f"{'AI-generated' if is_generated_aeroblade else 'Not AI-generated'} (Error: {error:.4f})"
                }
                # Only set final result if HuggingFace failed
                if 'is_ai_generated' not in final_result:
                    final_result = {
                        'is_ai_generated': is_generated_aeroblade,
                        'confidence': 1 - float(error),  # Invert error to get a confidence-like score
                        'method': 'AEROBLADE'
                    }
            except Exception as e:
                logger.error(f"Failed to analyze image with AEROBLADE: {str(e)}")
                result['aeroblade'] = {'error': f'Failed to analyze image with AEROBLADE: {str(e)}'}

        # Add the final result to the response
        result['final_result'] = final_result

        logger.info(f"Analysis result: {result}")
        return jsonify(result)

    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': f'Unexpected error: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)
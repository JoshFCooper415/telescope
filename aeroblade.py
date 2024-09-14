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

    def detect_aeroblade(self, image, threshold_definitely=0.001, threshold_probably=0.006, threshold_possibly=0.01):
        self.logger.info("Entered detect_aeroblade method")
        
        try:
            self.logger.info("About to compute reconstruction error")
            error = self.compute_reconstruction_error(image)
            self.logger.info(f"Reconstruction error: {error}")
        except Exception as e:
            self.logger.exception(f"Error in compute_reconstruction_error: {str(e)}")
            raise

        try:
            match error:
                case _ if error <= threshold_definitely:
                    result = "Definitely AI-generated"
                case _ if threshold_definitely < error <= threshold_probably:
                    result = "Probably AI-generated"
                case _ if threshold_probably < error <= threshold_possibly:
                    result = "Probably not AI-generated"
                case _:
                    result = "Not AI-generated"
            
            self.logger.info(f"AEROBLADE detection result: {result}, error={error}")
        except Exception as e:
            self.logger.exception(f"Error in match statement: {str(e)}")
            raise

        return result, (error/threshold_possibly)

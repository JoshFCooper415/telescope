import torch
import torchvision.transforms as transforms
from PIL import Image
import requests
from io import BytesIO
from lpips import LPIPS
from diffusers import AutoencoderKL

class AEROBLADE:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.lpips = LPIPS(net='vgg').to(self.device)
        self.lpips.eval()
        for param in self.lpips.parameters():
            param.requires_grad = False
        
        self.vae_v1 = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae").to(self.device)

        self.vae_v2 = AutoencoderKL.from_pretrained("stabilityai/stable-diffusion-2-1", subfolder="vae").to(self.device)

        self.vae_v3 = AutoencoderKL.from_pretrained("black-forest-labs/FLUX.1-dev", subfolder="vae").to(self.device)
        
        self.transform = transforms.Compose([
            transforms.CenterCrop(512),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    @torch.no_grad()
    def compute_reconstruction_error(self, image):
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        errors = []
        for vae in [self.vae_v1, self.vae_v2, self.vae_v3]:
            latents = vae.encode(image_tensor).latent_dist.sample()
            reconstructed = vae.decode(latents).sample
            
    
            reconstructed = (reconstructed / 2 + 0.5).clamp(0, 1)
            
        
            error = self.lpips.net.slice1(image_tensor - reconstructed).mean().item()
            errors.append(error)
        
        return min(errors)

    def detect(self, image, threshold=0.27): 
        error = self.compute_reconstruction_error(image)
        is_generated = error < threshold
        confidence = 1 - error
        return is_generated, confidence

def load_image_from_url(url):
    response = requests.get(url)
    return Image.open(BytesIO(response.content)).convert('RGB')

def main():
    detector = AEROBLADE()

    image_urls = [
        "https://cdn.pixabay.com/photo/2015/04/23/22/00/tree-736885_1280.jpg",
        "https://www.piclumen.com/wp-content/uploads/2024/08/ai-generated-female-with-wings-standing-on-the-woods.webp",
        "https://www.piclumen.com/wp-content/uploads/2024/08/ai-image.webp"
    ]

    for i, url in enumerate(image_urls):
        image = load_image_from_url(url)
        is_generated, confidence = detector.detect(image)
        
        print(f"Image {i+1}:")
        print(f"  AI-generated: {'Yes' if is_generated else 'No'}")
        print(f"  Confidence: {confidence:.4f}")
        print()

if __name__ == "__main__":
    main()
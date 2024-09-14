import torch
import torchvision.transforms as transforms
from lpips import LPIPS
from diffusers import AutoencoderKL
from datasets import load_dataset
from PIL import Image
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from tqdm import tqdm

class AEROBLADE:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.lpips = LPIPS(net='vgg', version='0.1').to(self.device)
        self.lpips.eval()
        
        self.aes = {
            'SD1': AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae").to(self.device),
            'SD2': AutoencoderKL.from_pretrained("stabilityai/stable-diffusion-2-1", subfolder="vae").to(self.device),

        }
        
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
        ])

    @torch.no_grad()
    def compute_reconstruction_error(self, image):
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        errors = []
        for ae in self.aes.values():
            latents = ae.encode(image_tensor).latent_dist.sample()
            reconstructed = ae.decode(latents).sample
            error = self.lpips(image_tensor, reconstructed).item()
            errors.append(error)
        
        return min(errors)

def process_dataset(detector, dataset, num_samples=1000):
    errors = []
    labels = []
    
    for split in ['test', 'train']:
        for i, item in tqdm(enumerate(dataset[split]), total=num_samples, desc=f"Processing {split} set"):
            if i >= num_samples:
                break
            image = item['image']
            error = detector.compute_reconstruction_error(image)
            errors.append(error)
            labels.append(item['label'])
    
    return np.array(errors), np.array(labels)

def plot_roc_curve(y_true, y_scores):
    # Print some statistics about the data
    print(f"Number of samples: {len(y_true)}")
    print(f"Number of positive samples: {np.sum(y_true)}")
    print(f"Number of negative samples: {len(y_true) - np.sum(y_true)}")
    print(f"Min score: {np.min(y_scores)}")
    print(f"Max score: {np.max(y_scores)}")
    print(f"Mean score: {np.mean(y_scores)}")
    print(f"Std score: {np.std(y_scores)}")

    # Check for NaN or infinity values
    if np.isnan(y_scores).any() or np.isinf(y_scores).any():
        print("Warning: y_scores contains NaN or infinity values")
        y_scores = np.nan_to_num(y_scores)

    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig('roc_curve.png')
    plt.close()
    
    return roc_auc

def main():
    print("Loading CIFAKE dataset...")
    ds = load_dataset("dragonintelligence/CIFAKE-image-dataset")
    
    print("Initializing AEROBLADE detector...")
    detector = AEROBLADE()
    
    print("Processing dataset...")
    errors, labels = process_dataset(detector, ds)
    
    print("Plotting ROC curve...")
    auc_score = plot_roc_curve(labels, errors)
    
    print(f"AUC Score: {auc_score:.4f}")

if __name__ == "__main__":
    main()
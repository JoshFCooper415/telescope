from binoculars import Binoculars
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc
from sklearn.utils import shuffle

BINOCULARS_MODEL_PERFORMER_NAME = "HuggingFaceTB/SmolLM-360M-Instruct"
BINOCULARS_MODEL_OBSERVER_NAME = "HuggingFaceTB/SmolLM-360M"

with open("hugging_face_auth_token.txt") as file:
    HUGGING_FACE_AUTH_TOKEN = file.readline()

def quantize_8bit(data):
    """Quantize the data to 8-bit."""
    min_val = np.min(data)
    max_val = np.max(data)
    scale = (max_val - min_val) / 255
    quantized = np.round((data - min_val) / scale).astype(np.uint8)
    return quantized, min_val, scale

def dequantize_8bit(quantized, min_val, scale):
    """Dequantize the 8-bit data back to original scale."""
    return (quantized.astype(np.float32) * scale) + min_val

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
    essay_dataset = pd.read_csv("validate_datasets/Essay_Dataset.csv").shuff

    text_dataset = essay_dataset["text"]
    is_ai_generated_dataset = essay_dataset["generated"]
   
    text_detector = Binoculars(BINOCULARS_MODEL_OBSERVER_NAME, BINOCULARS_MODEL_PERFORMER_NAME, HUGGING_FACE_AUTH_TOKEN)

    
    labels = []
    telescope_scores = []
    for text_data, is_ai_generated in tqdm(zip(text_dataset, is_ai_generated_dataset), total=len(text_dataset)):
        telescope_score = text_detector.compute_score(text_data, "cuda:0")
       
        print(f"is ai generated: {is_ai_generated}, score: {telescope_score}")
        labels.append(is_ai_generated)
        telescope_scores.append(telescope_score)
    
    # Convert lists to numpy arrays
    labels = np.array(labels)
    telescope_scores = np.array(telescope_scores)
    
    # Perform 8-bit quantization
    quantized_scores, min_val, scale = quantize_8bit(telescope_scores)
    
    # Dequantize for plotting and evaluation
    dequantized_scores = dequantize_8bit(quantized_scores, min_val, scale)
    
    plot_roc_curve(labels, dequantized_scores)

if __name__ == "__main__":
    main()
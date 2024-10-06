from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedModel, PreTrainedTokenizer, QuantoConfig
import quanto
import torch

from typing import Tuple



def load_model_and_tokenizer(model_path: str, hugging_face_auth_token: str = None, quantization_config = None) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    print(f"Loading tokenizer from {model_path}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, token = hugging_face_auth_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("Pad token set to EOS token: ", tokenizer.pad_token)
    print("Tokenizer loaded successfully")
    
    # Check CUDA availability
    if torch.cuda.is_available():
        print("CUDA is available. Using GPU.")
    else:
        print("CUDA is not available. Using CPU.")
    
    # Load the base model
    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        token=hugging_face_auth_token,
        quantization_config=quantization_config,
        device_map="auto",
        attn_implementation="flash_attention_2"
    )
    
    # Move model to the appropriate device
    print("Base model loaded successfully")
    
    return model, tokenizer


def get_hugging_face_auth_token(auth_token_filename: str):
    with open(auth_token_filename) as file:
        auth_token = file.readline()
        
    return auth_token

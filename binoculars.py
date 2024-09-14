import quanto.quantize
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedModel, PreTrainedTokenizer, QuantoConfig
import quanto
import torch

from typing import Tuple
import re

BINOCULARS_MODEL_PERFORMER_NAME = "google/gemma-2-2b-it"
BINOCULARS_MODEL_OBSERVER_NAME = "google/gemma-2-2b"

QUANTIZATION_CONFIG = QuantoConfig(weights="float8")

BINOCULARS_ACCURACY_THRESHOLD = 0.9015310749276843 
BINOCULARS_FPR_THRESHOLD = 0.8536432310785527 


class Binoculars:
    
    def __init__(self, observer_model_hf_name: str, performer_model_hf_name: str, hugging_face_api_token: str):
    
        self.performer_model, self.performer_tokenizer = self.load_model_and_tokenizer(performer_model_hf_name, hugging_face_api_token)
        self.observer_model, self.observer_tokenizer = self.load_model_and_tokenizer(observer_model_hf_name, hugging_face_api_token)
        
        quanto.quantize(self.performer_model, QUANTIZATION_CONFIG)
        quanto.quantize(self.observer_model, QUANTIZATION_CONFIG)
    
    
    def predict(
        self, reference_text: str,
        tokenizer: AutoTokenizer,
        performer_model: AutoModelForCausalLM,
        observer_model: AutoModelForCausalLM,
        device
    ):
        return (self.compute_log_perplexity(performer_model, tokenizer, reference_text, device)) / (
            self.compute_log_cross_perplexity(reference_text, tokenizer, observer_model, performer_model, device))
    
    
    def compute_log_perplexity(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, reference_text: str, device):
        
        reference_text_tokens = tokenizer(reference_text, return_tensors="pt").to(device)
        
        total_log_likelihood = 0
        word_count = 0

        context_tokens = tokenizer.encode("", return_tensors="pt").to(device).type(torch.int32)
        # print(tokenizer.encode("The", return_tensors="pt").to(device).type(torch.int32))
        for token in reference_text_tokens['input_ids'][0]:
            
            context_tokens = torch.cat([context_tokens, token.reshape(1, 1)], dim=-1)
            outputs = model(context_tokens)
    
            next_token_logits = outputs.logits[:, -1, :]
            next_tokens_logits_softmax = torch.softmax(next_token_logits, dim=-1)
            
            total_log_likelihood -= torch.log(next_tokens_logits_softmax[:,token])
            # print(total_log_likelihood)
            
            word_count += 1
            
        return total_log_likelihood / word_count
            
            
    def compute_log_cross_perplexity(
        self, reference_text: str,
        tokenizer: AutoTokenizer,
        observer_model: AutoModelForCausalLM,
        performer_model: AutoModelForCausalLM,
        device
        ):
        
        reference_text_tokens = tokenizer(reference_text, return_tensors="pt").to(device)
        eos_token_id = tokenizer.eos_token_id
        
        total_log_likelihood = 0
        word_count = 0

        context_tokens = tokenizer.encode("", return_tensors="pt").to(device).type(torch.int32)
        # print(tokenizer.encode("The", return_tensors="pt").to(device).type(torch.int32))
        for token in reference_text_tokens['input_ids'][0]:
            
            context_tokens = torch.cat([context_tokens, token.reshape(1, 1)], dim=-1)
            
            performer_outputs = performer_model(context_tokens)
            observer_outputs = observer_model(context_tokens)
    
            performer_next_token_logits = performer_outputs.logits[:, -1, :]
            observer_next_token_logits = observer_outputs.logits[:, -1, :]
            
            # print(observer_outputs.logits)
            
            performer_next_tokens_logits_softmax = torch.softmax(performer_next_token_logits, dim=-1)
            observer_next_token_logits_softmax = torch.softmax(observer_next_token_logits, dim=-1)
            
            # print(performer_next_tokens_logits_softmax)
            # print(torch.log(observer_next_token_logits_softmax))
            total_log_likelihood -= torch.dot(performer_next_tokens_logits_softmax[-1], torch.log(observer_next_token_logits_softmax[-1])) 
            # print(total_log_likelihood)
            
            word_count += 1
            
        return total_log_likelihood / word_count
        
        
    
    
    def load_model_and_tokenizer(self, model_path: str, hugging_face_auth_token: str = None) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
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
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # Move model to the appropriate device
        print("Base model loaded successfully")
        
        return model, tokenizer


def get_hugging_face_auth_token(auth_token_filename: str):
    with open(auth_token_filename) as file:
        auth_token = file.readline()
        
    return auth_token
    
    

if __name__ == "__main__":
    hugging_face_auth_token = get_hugging_face_auth_token("hugging_face_auth_token.txt")
    binoculars = Binoculars(BINOCULARS_MODEL_OBSERVER_NAME, BINOCULARS_MODEL_PERFORMER_NAME, hugging_face_auth_token)
    
    print(binoculars.predict(
        "Central to Hitler's ideology was the belief in racial purity and the superiority of the so-called Aryan race. He sought to create a homogeneous, racially pure German society by promoting policies that encouraged the reproduction of Aryan individuals while eliminating those he deemed racially inferior. This led to the promotion of eugenics, a pseudo-scientific movement aimed at improving the genetic quality of the human population through selective breeding. The Nazis sterilized thousands of people with disabilities", 
        binoculars.performer_tokenizer, 
        binoculars.observer_model, 
        binoculars.performer_model,
        "cuda:0",
    ))
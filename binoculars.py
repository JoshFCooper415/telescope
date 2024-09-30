import quanto.quantize
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedModel, PreTrainedTokenizer, QuantoConfig
from transformers import BitsAndBytesConfig
import quanto
import torch


from typing import Tuple, List
import re
import numpy as np

# BINOCULARS_MODEL_PERFORMER_NAME = "google/gemma-2-2b-it"
# BINOCULARS_MODEL_OBSERVER_NAME = "google/gemma-2-2b"

BINOCULARS_MODEL_PERFORMER_NAME = "HuggingFaceTB/SmolLM-360M-Instruct"
BINOCULARS_MODEL_OBSERVER_NAME = "HuggingFaceTB/SmolLM-360M"


QUANTO_CONFIG = QuantoConfig(weights="int4")
BITS_AND_BYTES_CONFIG = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)

class Binoculars:
    
    def __init__(self, observer_model_hf_name: str, performer_model_hf_name: str, hugging_face_api_token: str):
    
        self.performer_model, self.performer_tokenizer = self.load_model_and_tokenizer(performer_model_hf_name, hugging_face_api_token, BITS_AND_BYTES_CONFIG)
        self.observer_model, self.observer_tokenizer = self.load_model_and_tokenizer(observer_model_hf_name, hugging_face_api_token, BITS_AND_BYTES_CONFIG)


    def predict(self, reference_text, device, threshold_possibly=5.3, threshold_probably=5.7, threshold_definitely=6.2) -> Tuple[str, float]:
        telescope_score = self.compute_score(reference_text, device)
        
        result_text = ""
        if telescope_score >= threshold_definitely:
            result_text = "Definitely AI-generated"
        elif threshold_definitely > telescope_score >= threshold_probably:
            result_text = "Probably AI-generated"
        elif threshold_probably > telescope_score >= threshold_possibly:
            result_text = "Probably not AI-generated"
        else:
            result_text = "Not AI-generated"
            
        return result_text, telescope_score
        
        
    def compute_score(self, reference_text: str, device) -> float:
        score = self.compute_telescope_perplexity(reference_text, self.performer_model, self.observer_model, self.performer_tokenizer, device)

        return score.cpu()[0]
    
    
    # def compute_score_batched(self, batched_reference_text: List[str], device) -> float:
    #     score = self.compute_telescope_perplexity_batched(batched_reference_text, self.performer_model, self.observer_model, self.performer_tokenizer, device)

    #     return score.cpu()[0]


    @torch.inference_mode()
    def compute_telescope_perplexity(
        self, 
        reference_text: str,
        performer_model: AutoModelForCausalLM,
        observer_model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        device
        ):
        
        reference_text_tokens = tokenizer(reference_text, return_tensors="pt").to(device)
        context_tokens = tokenizer.encode("", return_tensors="pt").to(device).type(torch.int32)

        total_cross_entropy_cross_perplexity = 0
        total_cross_entropy_normal_perplexity = 0

        for token in reference_text_tokens['input_ids'][0]:
            
            context_tokens = torch.cat([context_tokens, token.reshape(1, 1)], dim=-1)
            
            performer_outputs = performer_model(context_tokens)
            observer_outputs = observer_model(context_tokens)
    
            performer_next_token_logits = performer_outputs.logits[:, -1, :]
            observer_next_token_logits = observer_outputs.logits[:, -1, :]

            performer_next_tokens_logits_softmax = torch.softmax(performer_next_token_logits, dim=-1)
            observer_next_token_logits_softmax = torch.softmax(observer_next_token_logits, dim=-1)
            
            total_cross_entropy_cross_perplexity -= torch.matmul(performer_next_tokens_logits_softmax, torch.log(observer_next_token_logits_softmax).T) 
            total_cross_entropy_normal_perplexity -= torch.log(performer_next_tokens_logits_softmax[:,token])
            
            
        return total_cross_entropy_normal_perplexity/  total_cross_entropy_cross_perplexity
    
 
 
 
 
    def compute_telescope_perplexity_batched(encoding: transformers.BatchEncoding,
                                   logits: torch.Tensor,
                                   median: bool = False,
                                   temperature: float = 1.0
                                   ):
        
        ce_loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
        
        shifted_logits = logits[..., :-1, :].contiguous() / temperature
        shifted_labels = encoding.input_ids[..., 1:].contiguous()
        shifted_attention_mask = encoding.attention_mask[..., 1:].contiguous()

        if median:
            ce_nan = (ce_loss_fn(shifted_logits.transpose(1, 2), shifted_labels).
                    masked_fill(~shifted_attention_mask.bool(), float("nan")))
            ppl = np.nanmedian(ce_nan.cpu().float().numpy(), 1)

        else:
            ppl = (ce_loss_fn(shifted_logits.transpose(1, 2), shifted_labels) *
                shifted_attention_mask).sum(1) / shifted_attention_mask.sum(1)
            ppl = ppl.to("cpu").float().numpy()

        return ppl
    
    @torch.inference_mode()
    def compute_telescope_perplexity_batched(self, 
        batched_reference_text: List[str],
        performer_model: AutoModelForCausalLM,
        observer_model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        device
        ):
        
        batch_size = len(batched_reference_text)
        encodings = self.tokenizer(
            batched_reference_text,
            return_tensors="pt",
            padding="longest" if batch_size > 1 else False,
            truncation=True,
            max_length=self.max_token_observed,
            return_token_type_ids=False
        ).to(self.observer_model.device)
        
        
        observer_logits = self.observer_model(**encodings.to("auto")).logits
        performer_logits = self.performer_model(**encodings.to("auto")).logits
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        ppl = self.compute_telescope_perplexity_batched(encodings, performer_logits)
        x_ppl = entropy(observer_logits.to(DEVICE_1), performer_logits.to(DEVICE_1),
                        encodings.to(DEVICE_1), self.tokenizer.pad_token_id)
        binoculars_scores = ppl / x_ppl
        return ppl
    
    
    def load_model_and_tokenizer(self, model_path: str, hugging_face_auth_token: str = None, quantization_config = None) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
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
            # load_in_4bit=True,
            attn_implementation="flash_attention_2"
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
    
    with open("binoculars_test_prompt.txt") as file:
        SENTENCE = "\n".join(file.readlines())
        
    # SENTENCE = "Rose is a flower so beautiful that it has invoked inspiration in several artists and poets. Children are familiar with the rose and other such flowers right from toddlerhood when they took strolls in the garden, to the time they started enjoying picture books, to learning the alphabet 'R' for rose. The flower may have been a part of their home décor, or a gift they gave someone on an occasion."
    is_ai_generated, score = binoculars.predict(SENTENCE, "cuda:0")
    print(f"is ai generated: {is_ai_generated}")
    print(f"score: {score}")
    
    
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedModel, PreTrainedTokenizer, BatchEncoding
from transformers import BitsAndBytesConfig, QuantoConfig
import torch

import time
from typing import Tuple, List, Union

from utils import load_model_and_tokenizer, get_hugging_face_auth_token, create_attention_mask
from metrics import perplexity, entropy


# BINOCULARS_MODEL_PERFORMER_NAME = "google/gemma-2-2b-it"
# BINOCULARS_MODEL_OBSERVER_NAME = "google/gemma-2-2b"

BINOCULARS_ACCURACY_THRESHOLD = 0.9015310749276843  # optimized for f1-score
BINOCULARS_FPR_THRESHOLD = 0.8536432310785527  # optimized for low-fpr [chosen at 0.01%]

PERFORMER_MODEL_NAME = "HuggingFaceTB/SmolLM-360M-Instruct"
OBSERVER_MODEL_NAME = "HuggingFaceTB/SmolLM-360M"

BITS_AND_BYTES_CONFIG = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)



class Telescope:
    
    def __init__(self, observer_model_hf_name: str, performer_model_hf_name: str, hugging_face_api_token: str):
    
        self.performer_model, self.performer_tokenizer = load_model_and_tokenizer(performer_model_hf_name, hugging_face_api_token, BITS_AND_BYTES_CONFIG)
        self.observer_model, self.observer_tokenizer = load_model_and_tokenizer(observer_model_hf_name, hugging_face_api_token, BITS_AND_BYTES_CONFIG)


    def predict(self, reference_text: str, device: torch.device, threshold_possibly=5.3, threshold_probably=5.7, threshold_definitely=6.2) -> Tuple[str, float]:
        """
        Computes the telescope score (optimized for single inference) 
        and returns a piece of text describing how sure the model is of whether the given text is AI generated based on the given thresholds

        Args:
            reference_text (str): The text to predict whether it is AI generated or not
            device (torch.device): The device to perform all of the computations (this is almost always some sort of gpu)
            threshold_possibly (float, optional):  Defaults to 5.3.
            threshold_probably (float, optional): Defaults to 5.7.
            threshold_definitely (float, optional): Defaults to 6.2.

        Returns:
            Tuple[str, float]: Returns a tuple of (result text, telescope score).
            
            The result text is just a string describing how likely the model thinks the given text is AI generated
            and the telescope score is the raw score from the model.
        """
        telescope_score = self.compute_score(reference_text, device, use_binoculars=False)
        
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
        
        
    def compute_score(self, reference_text: Union[str, List[str]], device: torch.device, use_binoculars=False, batch_size = 1) -> float:
        
        if use_binoculars == True:
            score = self.compute_binoculars_perplexity(reference_text, self.performer_model, self.observer_model, self.performer_tokenizer, device)
        else:
            score = self.compute_telescope_perplexity_unbatched(reference_text, self.performer_model, self.observer_model, self.performer_tokenizer, device)
            # score = self.compute_telescope_perplexity_batched(
            #     reference_text, 
            #     self.performer_model, 
            #     self.observer_model, 
            #     self.performer_tokenizer, 
            #     batch_size=batch_size, 
            #     device=device
            # )

        return score
 
    @torch.inference_mode()
    def compute_telescope_perplexity_causal_mask(
        self, 
        reference_text: str,
        performer_model: AutoModelForCausalLM,
        observer_model: AutoModelForCausalLM,
        tokenizer: PreTrainedTokenizer,
        batch_size: int = 1,
        number_of_tokens_to_skip: int = 1,
        device: torch.device = torch.device("cpu"),
        ):
        
        tokenizer.padding_side = "left"
        tokenizer.pad_token = tokenizer.eos_token
        
        reference_text_encodings = tokenizer(reference_text, return_tensors="pt").to(device)

        attention_mask_size = reference_text_encodings["attention_mask"].shape[1]
        attention_mask = torch.tril(torch.ones(attention_mask_size, attention_mask_size, device=device)).unsqueeze(0)
        reference_text_encodings["attention_mask"] = attention_mask
        
        performer_outputs = performer_model(**reference_text_encodings)
        observer_outputs = observer_model(**reference_text_encodings)
        
        print("got here")
        
        total_cross_entropy_cross_perplexity = 0
        total_cross_entropy_normal_perplexity = 0
        
        # Loop over each item in the chunk and calculate the telescope score for that item
        for index in range(0, performer_outputs.logits.shape[0]):
            performer_next_token_logits = performer_outputs.logits[index, -1, :]
            observer_next_token_logits = observer_outputs.logits[index, -1, :]
                            
            performer_next_tokens_logits_softmax = torch.softmax(performer_next_token_logits, dim=-1)
            observer_next_token_logits_softmax = torch.softmax(observer_next_token_logits, dim=-1)
            
            # print(f"index: {index}, cross_perp: {torch.matmul(performer_next_tokens_logits_softmax, torch.log(observer_next_token_logits_softmax).T)}, normal perp: {torch.log(performer_next_tokens_logits_softmax[reference_text_tokens[index+1]])}")
            total_cross_entropy_cross_perplexity -= torch.matmul(performer_next_tokens_logits_softmax, torch.log(observer_next_token_logits_softmax).reshape(1, -1).T) 
            total_cross_entropy_normal_perplexity -= torch.log(performer_next_tokens_logits_softmax[reference_text_tokens[index+1]])

        print(total_cross_entropy_normal_perplexity/total_cross_entropy_cross_perplexity)
        return total_cross_entropy_normal_perplexity.cpu() /total_cross_entropy_cross_perplexity.cpu()
        
        
        
        
        
    @torch.inference_mode()
    def compute_telescope_perplexity_batched(
        self, 
        reference_text: str,
        performer_model: AutoModelForCausalLM,
        observer_model: AutoModelForCausalLM,
        tokenizer: PreTrainedTokenizer,
        batch_size: int = 1,
        number_of_tokens_to_skip: int = 1,
        device: torch.device = torch.device("cpu"),
        ):
        
    
        tokenizer.padding_side = "left"
        tokenizer.pad_token = tokenizer.eos_token
        
        # prepare text that is split over every token
        reference_text_tokens: List[int] = []
        reference_text_tokens = tokenizer(reference_text)["input_ids"]
        
        reference_text_split_by_tokens: List[str] = []
        for token in reference_text_tokens:
            reference_text_split_by_tokens.append(tokenizer.decode(token))
        
        
        # prepare list of all contexts. This means that we want the reference text contexts to look like the following for the sequence "I like pizza":
        # ["I", "I like", "I like pizza"]
        reference_text_contexts: List[str] = []
        temp_reference_text_history = ""
        for index, token in enumerate(reference_text_split_by_tokens):
            if index >= number_of_tokens_to_skip:
                reference_text_contexts.append(temp_reference_text_history)
                
            temp_reference_text_history += token
            
        
        # Cut the list of contexts into chunks of size equal to the batch size, these chunks will be processed together at inference
        reference_text_batched: List[List[str]] = []
        temp_array = []
        for index, context_string in enumerate(reference_text_contexts):
            if (index % batch_size) == 0 and index != 0:
                reference_text_batched.append(temp_array)
                temp_array = []
            
            temp_array.append(context_string)
        reference_text_batched.append(temp_array)
        
        total_cross_entropy_cross_perplexity = 0
        total_cross_entropy_normal_perplexity = 0
        
        
        for chunk_index, reference_text_chunk in enumerate(reference_text_batched):
            
            # Tokenize each chunk
            reference_text_chunk_encoding: BatchEncoding = tokenizer(reference_text_chunk, return_tensors="pt", padding=True).to(device)
            # print(reference_text_chunk_encoding["attention_mask"])

            # Compute the logits from the model
            performer_outputs = performer_model(**reference_text_chunk_encoding)
            observer_outputs = observer_model(**reference_text_chunk_encoding)


            # Loop over each item in the chunk and calculate the telescope score for that item
            for index in range(0, performer_outputs.logits.shape[0]):
                performer_next_token_logits = performer_outputs.logits[index, -1, :]
                observer_next_token_logits = observer_outputs.logits[index, -1, :]
                                
                performer_next_tokens_logits_softmax = torch.softmax(performer_next_token_logits, dim=-1)
                observer_next_token_logits_softmax = torch.softmax(observer_next_token_logits, dim=-1)
                
                # print(f"index: {index}, cross_perp: {torch.matmul(performer_next_tokens_logits_softmax, torch.log(observer_next_token_logits_softmax).T)}, normal perp: {torch.log(performer_next_tokens_logits_softmax[reference_text_tokens[index+1]])}")
                total_cross_entropy_cross_perplexity -= torch.matmul(performer_next_tokens_logits_softmax, torch.log(observer_next_token_logits_softmax).reshape(1, -1).T) 
                total_cross_entropy_normal_perplexity -= torch.log(performer_next_tokens_logits_softmax[reference_text_tokens[index + chunk_index*batch_size]])

                print(f"index: {index}, cross: {torch.matmul(performer_next_tokens_logits_softmax, torch.log(observer_next_token_logits_softmax).T) }, normal: {torch.log(performer_next_tokens_logits_softmax[reference_text_tokens[index+1 + chunk_index*batch_size]])}")

                del observer_next_token_logits_softmax, performer_next_tokens_logits_softmax, performer_next_token_logits, observer_next_token_logits
                torch.cuda.empty_cache() 

            del performer_outputs, observer_outputs, reference_text_chunk_encoding
            torch.cuda.empty_cache() 

        # print(total_cross_entropy_normal_perplexity/  total_cross_entropy_cross_perplexity)
        return total_cross_entropy_normal_perplexity.cpu()/  total_cross_entropy_cross_perplexity.cpu()
        
        
        
    @torch.inference_mode()
    def get_model_output(self, model, input):
        return model(input)
        
        
    def compute_telescope_perplexity_unbatched(
        self, 
        reference_text: str,
        performer_model: AutoModelForCausalLM,
        observer_model: AutoModelForCausalLM,
        tokenizer: PreTrainedTokenizer,
        device: torch.device
        ):

        reference_text_batch_encoding: BatchEncoding = tokenizer(reference_text, return_tensors="pt", padding=True, truncation=True).to(device)
        reference_text_tokens = reference_text_batch_encoding["input_ids"]
 
        NUMBER_OF_TOKENS_TO_SKIP = 0
    
        total_cross_entropy_cross_perplexity = 0
        total_cross_entropy_normal_perplexity = 0
        

        reference_text_tokens = reference_text_tokens[0]
        for token_index, token in enumerate(reference_text_tokens):
            if token_index < NUMBER_OF_TOKENS_TO_SKIP: continue
            
            context_tokens = reference_text_tokens[:(token_index+1)].reshape(1, -1)
            
            performer_outputs = self.get_model_output(performer_model, context_tokens)
            observer_outputs = self.get_model_output(observer_model, context_tokens)
    
            performer_next_token_logits = performer_outputs.logits[:, -1, :]
            observer_next_token_logits = observer_outputs.logits[:, -1, :]

            performer_next_tokens_logits_softmax = torch.softmax(performer_next_token_logits, dim=-1)
            observer_next_token_logits_softmax = torch.softmax(observer_next_token_logits, dim=-1)
            
            # print(f"index: {token_index}, cross_perp: {torch.matmul(performer_next_tokens_logits_softmax, torch.log(observer_next_token_logits_softmax).T)}, normal perp: {torch.log(performer_next_tokens_logits_softmax[:, token])}")

            total_cross_entropy_cross_perplexity -= torch.matmul(performer_next_tokens_logits_softmax, torch.log(observer_next_token_logits_softmax).T) 
            total_cross_entropy_normal_perplexity -= torch.log(performer_next_tokens_logits_softmax[:, token])
            
            print(f"index: {token_index}, cross: {torch.matmul(performer_next_tokens_logits_softmax, torch.log(observer_next_token_logits_softmax).T) }, normal: {torch.log(performer_next_tokens_logits_softmax[:, token])}")

            del observer_next_token_logits_softmax
            del performer_next_tokens_logits_softmax
            del performer_next_token_logits
            del observer_next_token_logits
            del context_tokens
            torch.cuda.empty_cache() 
            
        print(total_cross_entropy_normal_perplexity/  total_cross_entropy_cross_perplexity)
        return total_cross_entropy_normal_perplexity/  total_cross_entropy_cross_perplexity
    
 
        
    @torch.inference_mode()
    def compute_binoculars_perplexity_old(
        self, 
        reference_text: str,
        performer_model: AutoModelForCausalLM,
        observer_model: AutoModelForCausalLM,
        tokenizer: PreTrainedTokenizer,
        device: torch.device
        ):
        
        # tokenizer.padding_side = "left"
        # tokenizer.pad_token = tokenizer.eos_token

        reference_text_batch_encoding: BatchEncoding = tokenizer(reference_text, return_tensors="pt", padding=True, truncation=True).to(device)
        reference_text_tokens = reference_text_batch_encoding["input_ids"]
        number_of_words_in_reference_text = reference_text_tokens.shape[1]

        # total_cross_entropy_cross_perplexity = 0
        # total_cross_entropy_normal_perplexity = 0
        
        # BATCH_SIZE = 4
        # BATCH_COUNT = int(np.ceil(number_of_words_in_reference_text/BATCH_SIZE))
        NUMBER_OF_TOKENS_TO_SKIP = 1
        
        # for batch_index in range(BATCH_COUNT):
        #     batch_starting_index = batch_index * BATCH_SIZE + NUMBER_OF_TOKENS_TO_SKIP
        #     batch_ending_index = min(number_of_words_in_reference_text, batch_starting_index + BATCH_SIZE)

        #     if (batch_starting_index == batch_ending_index): continue
            
        #     # calculate attention mask for each token (context is all the same which is the entire sequence)
        #     attention_masks = []
        #     for index in range(batch_starting_index, batch_ending_index):
        #         current_token_attention_mask = create_attention_mask(total_length=number_of_words_in_reference_text, number_of_words_to_include=index, device=device)
        #         attention_masks.append(current_token_attention_mask)

        #     attention_masks = torch.concat(attention_masks, dim=0)
        #     reference_text_batch_encoding["input_ids"] = torch.concat([reference_text_tokens for i in range(batch_starting_index, batch_ending_index)], dim=0)
        #     reference_text_batch_encoding['attention_mask'] = attention_masks
            
        #     performer_outputs = performer_model(**reference_text_batch_encoding)
        #     observer_outputs = observer_model(**reference_text_batch_encoding)

        #     # Loop over each item in the batch
        #     for index in range(0, batch_ending_index - batch_starting_index):
        #         # print(f"index: {index}, batch start: {batch_starting_index}")
        #         # batch_starting_index + index
        #         performer_next_token_logits = performer_outputs.logits[index, batch_starting_index + index, :].reshape(1, -1)
        #         observer_next_token_logits = observer_outputs.logits[index, batch_starting_index + index, :].reshape(1, -1)
                                
        #         performer_next_tokens_logits_softmax = torch.softmax(performer_next_token_logits, dim=-1)
        #         observer_next_token_logits_softmax = torch.softmax(observer_next_token_logits, dim=-1)
                
        #         print(f"index: {batch_starting_index + index}, cross_perp: {torch.matmul(performer_next_tokens_logits_softmax, torch.log(observer_next_token_logits_softmax).T)}, normal perp: {torch.log(performer_next_tokens_logits_softmax[0][reference_text_tokens[0][batch_starting_index + index]])}")
        #         total_cross_entropy_cross_perplexity -= torch.matmul(performer_next_tokens_logits_softmax, torch.log(observer_next_token_logits_softmax).T) 
        #         total_cross_entropy_normal_perplexity -= torch.log(performer_next_tokens_logits_softmax[0][reference_text_tokens[0][batch_starting_index + index]])
        
        
        # print(total_cross_entropy_normal_perplexity/  total_cross_entropy_cross_perplexity)
        # return total_cross_entropy_normal_perplexity/  total_cross_entropy_cross_perplexity
            
            
        # for batch_index in range(BATCH_COUNT):
        #     context_token_batch = reference_text_tokens[batch_index::BATCH_COUNT]
            
        #     for i, token in enumerate(reference_text_tokens[batch_index::BATCH_COUNT]):
        #         token_index = batch_index + i * BATCH_COUNT
        #         print(token_index)
        #         print(token)
        #         if token_index < NUMBER_OF_TOKENS_TO_SKIP: continue
                
        #         # THIS LINE IS A MISTAKE, THIS LINE SHOULD GO AT THE VERY END, BUT IT WORKS!!!!
        #         # context_tokens = torch.cat([context_tokens, token.reshape(1, 1)], dim=-1)
                
        #         # THIS LINE IS A BETTER WAY OF SAYING THE LAST LINE BUT IT IS STILL WRONG
        #         context_tokens = reference_text_tokens[:(token_index+1)].reshape(1, -1)
        #         print(context_tokens)
        #         print()
                
        #         # context = transformers.BatchEncoding({'input_ids': reference_text_tokens, 'attention_mask': torch.tensor([[1 if i <= token_index else 0 for i in range(len(reference_text_tokens))],])})
                
        #         # THIS IS RIGHT BUT NUMBER_OF_TOKENS_TO_SKIP MUST BE GREATER THAN OR EQUAL TO 1
        #         # context_tokens = reference_text_tokens[:(token_index+1)].reshape(1, -1)
                
        #         performer_outputs = performer_model()
        #         observer_outputs = observer_model()

                
        #         performer_next_token_logits = performer_outputs.logits[:, -1, :]
        #         observer_next_token_logits = observer_outputs.logits[:, -1, :]

        #         performer_next_tokens_logits_softmax = torch.softmax(performer_next_token_logits, dim=-1)
        #         observer_next_token_logits_softmax = torch.softmax(observer_next_token_logits, dim=-1)
                
        #         total_cross_entropy_cross_perplexity -= torch.matmul(performer_next_tokens_logits_softmax, torch.log(observer_next_token_logits_softmax).T) 
        #         total_cross_entropy_normal_perplexity -= torch.log(performer_next_tokens_logits_softmax[:, token])
        
        # Unbatched Implementation
        total_cross_entropy_cross_perplexity = 0
        total_cross_entropy_normal_perplexity = 0
        

        reference_text_tokens = reference_text_tokens[0]
        for token_index, token in enumerate(reference_text_tokens):
            if token_index < NUMBER_OF_TOKENS_TO_SKIP: continue
            
            # THIS LINE IS A MISTAKE, THIS LINE SHOULD GO AT THE VERY END, BUT IT WORKS!!!!
            # context_tokens = torch.cat([context_tokens, token.reshape(1, 1)], dim=-1)
            
            # THIS LINE IS A BETTER WAY OF SAYING THE LAST LINE BUT IT IS STILL WRONG
            # context_tokens = reference_text_tokens[:(token_index+1)].reshape(1, -1)
            
            # THIS LINE IS RIGHT!
            context_tokens = reference_text_tokens[:(token_index)].reshape(1, -1)
            
            # THIS IS RIGHT BUT NUMBER_OF_TOKENS_TO_SKIP MUST BE GREATER THAN OR EQUAL TO 1
            # context_tokens = reference_text_tokens[:(token_index+1)].reshape(1, -1)
            
            performer_outputs = performer_model(context_tokens)
            observer_outputs = observer_model(context_tokens)
    
            performer_next_token_logits = performer_outputs.logits[:, -1, :]
            observer_next_token_logits = observer_outputs.logits[:, -1, :]

            performer_next_tokens_logits_softmax = torch.softmax(performer_next_token_logits, dim=-1)
            observer_next_token_logits_softmax = torch.softmax(observer_next_token_logits, dim=-1)
            
            print(f"index: {token_index}, cross_perp: {torch.matmul(performer_next_tokens_logits_softmax, torch.log(observer_next_token_logits_softmax).T)}, normal perp: {torch.log(performer_next_tokens_logits_softmax[:, token])}")

            total_cross_entropy_cross_perplexity -= torch.matmul(performer_next_tokens_logits_softmax, torch.log(observer_next_token_logits_softmax).T) 
            total_cross_entropy_normal_perplexity -= torch.log(performer_next_tokens_logits_softmax[:, token])
        
        print("hi")
        print(total_cross_entropy_normal_perplexity/  total_cross_entropy_cross_perplexity)
        return total_cross_entropy_normal_perplexity/  total_cross_entropy_cross_perplexity
    
    
    
    @torch.inference_mode()
    def compute_binoculars_perplexity(
        self, 
        batched_reference_text: str,
        performer_model: AutoModelForCausalLM,
        observer_model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        device
        ):
        """Paper implementation of binoculars for comparison"""
        
        BATCH_SIZE = 1
        encodings = tokenizer(
            batched_reference_text,
            return_tensors="pt",
            padding="longest" if BATCH_SIZE > 1 else False,
            truncation=True,
            return_token_type_ids=False
        ).to(device)
        
        # print(encodings)
        
        observer_logits = observer_model(**encodings).logits
        performer_logits = performer_model(**encodings).logits
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        ppl = perplexity(encodings, performer_logits)
        x_ppl = entropy(observer_logits.to(device), performer_logits.to(device),
                        encodings.to(device), tokenizer.pad_token_id)
        binoculars_scores = ppl / x_ppl
        # print(binoculars_scores)
        return binoculars_scores

    
    
    
    
    

if __name__ == "__main__":
    hugging_face_auth_token = get_hugging_face_auth_token("hugging_face_auth_token.txt")
    binoculars = Telescope(OBSERVER_MODEL_NAME, PERFORMER_MODEL_NAME, hugging_face_auth_token)
    
    with open("test_prompt.txt") as file:
        SENTENCE = "\n".join(file.readlines())
        
    # SENTENCE = "Rose is a flower so beautiful that it has invoked inspiration in several artists and poets. Children are familiar with the rose and other such flowers right from toddlerhood when they took strolls in the garden, to the time they started enjoying picture books, to learning the alphabet 'R' for rose. The flower may have been a part of their home décor, or a gift they gave someone on an occasion."
    # is_ai_generated, score = binoculars.predict(SENTENCE, "cuda:0")
    # print(f"is ai generated: {is_ai_generated}")
    # print(f"score: {score}")
    
    start_time = time.time()
    score = binoculars.compute_score(SENTENCE, "cuda:0", batch_size=16, use_binoculars=False)
    print(f"score: {score}")
    print(f"time taken: {time.time() - start_time}")
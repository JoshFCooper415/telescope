import warnings
import torch
from transformers import TrainingArguments, Trainer
from torch.utils.data import ConcatDataset
from datasets import Dataset
from datasets import load_dataset  # Add this import
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from datasets import load_dataset

from utils import load_model_and_tokenizer


MODEL_TO_FINETUNE = "HuggingFaceTB/SmolLM-360M"
# MODEL_TO_FINETUNE = "HuggingFaceTB/SmolLM-360M-Instruct"
SAVE_NAME = "SmolLM-360M-LORA"

# FINETUNE_DATASET = "ise-uiuc/Magicoder-Evol-Instruct-110K"
# FINETUNE_DATASET = "bigcode/starcoderdata"
FINETUNE_DATASET = "iamtarun/code_instructions_120k_alpaca"


class FinetuneDataset(Dataset):
    def __init__(self, tokenizer, split='train', max_length=512, val_split=0.1):
        self.tokenizer = tokenizer
        self.max_length = max_length

        full_dataset = load_dataset(FINETUNE_DATASET, split="train")
        full_dataset = full_dataset.filter(lambda x: len(x['instruction']) > 0 and len(x['output']) > 0)
        
        # Split the dataset
        self._data = full_dataset
        # split_dataset = full_dataset.train_test_split(test_size=val_split)
        # self.data = split_dataset['train'] if split == 'train' else split_dataset['test']
       
       
    def __len__(self):
        return len(self._data)
   
    def __getitem__(self, idx):
        instruction = self._data[idx]['instruction']
        output = self._data[idx]['output']
        text = f"Instruction: {instruction}\nResponse: {output}"

        encoding = self.tokenizer(text, max_length=int(self.max_length), truncation=True, padding='max_length', return_tensors='pt')
        return encoding['input_ids'].squeeze(), encoding['attention_mask'].squeeze()



def main():
    warnings.filterwarnings("ignore", message="torch.utils.checkpoint: the use_reentrant parameter")

    torch.backends.cudnn.benchmark = True
    torch.cuda.empty_cache()

    with open("hugging_face_auth_token.txt") as file:
        HUGGING_FACE_AUTH_TOKEN = file.readline()

    print("Starting to load model and tokenizer...")
    
    model, tokenizer = load_model_and_tokenizer(MODEL_TO_FINETUNE, HUGGING_FACE_AUTH_TOKEN)
    
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["gate_proj", "down_proj", "up_proj", "q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)


    finetune_dataset_train = FinetuneDataset(tokenizer, max_length="1028", split="train")
    # finetune_dataset_validate = FinetuneDataset(tokenizer, max_length="1028", split="validate")
    
    training_args = TrainingArguments(
        output_dir=f"./{SAVE_NAME}",
        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=16,
        learning_rate=5e-5,
        warmup_steps=300,
        weight_decay=0.01,
        logging_steps=500,
        save_steps=500,
        save_total_limit=2,
        remove_unused_columns=False,
        fp16=True,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit"
    )

    print("Setting up trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=finetune_dataset_train,
    )

    try:
        print("Starting training...")
        trainer.train()
    except Exception as e:
        print(f"Error during training: {e}")
        raise

    print("Saving LoRA adapters...")
    model.save_pretrained(f"./{SAVE_NAME}")
    tokenizer.save_pretrained(f"./{SAVE_NAME}")
    print("Training completed and LoRA adapters saved.")

if __name__ == "__main__":
    main()
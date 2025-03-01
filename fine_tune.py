from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import Dataset
import torch
import os

# 1. Load Model and Tokenizer
model_name = "microsoft/Phi-3-mini-4k-instruct"
print(f"Loading model: {model_name}") # Debug print

if torch.cuda.is_available():
    device = "cuda"
    print("CUDA is available. Using GPU for fine-tuning.") # Debug print
    print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}") # Debug print device name
    print(f"CUDA Device Count: {torch.cuda.device_count()}") # Debug print device count
    print(f"CUDA Version: {torch.version.cuda}") # Debug print CUDA version
    # print(f"cuDNN Version: {torch.backends.cuda.version()}") # Debug print cuDNN version
    print(f"Is CUDA Built: {torch.backends.cuda.is_built()}") # Debug print if CUDA is built
    print(f"Current CUDA Device: {torch.cuda.current_device()}") # Debug print current device
    try:
        torch.tensor([1, 2, 3], device=device)
        print("CUDA initialization successful.") # Debug print
    except Exception as e:
        print(f"CUDA initialization failed: {e}") # Debug print

else:
    device = "cpu"
    print("CUDA is not available. Using CPU for fine-tuning.") # Debug print

model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, use_cache=False) # device_map="auto" will use GPU if available
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token # Set pad token to EOS token

# 2. Prepare Dataset
train_file_path = "fine_tune_slm/training_data.txt"
print(f"Loading dataset from: {train_file_path}") # Debug print
with open(train_file_path, "r", encoding="utf-8") as f:
    text_data = f.read()

train_dataset = Dataset.from_dict({"text": [text_data]}) # Simple dataset creation

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128) # Reduced max_length to reduce memory

print("Tokenizing dataset...") # Debug print
tokenized_datasets = train_dataset.map(tokenize_function, batched=True)
print("Dataset tokenized.") # Debug print

# 3. Define Training Arguments
training_args = TrainingArguments(
    output_dir="./fine_tune_slm/results",
    num_train_epochs=1,        # Just 1 epoch for a quick test run
    per_device_train_batch_size=1, # Reduced batch size to minimize memory usage
    gradient_accumulation_steps=4, # Effective batch size = batch_size * gradient_accumulation_steps
    warmup_steps=100,
    weight_decay=0.01,
    logging_dir="./fine_tune_slm/logs", # Log directory
    logging_steps=10,
    evaluation_strategy="no",     # No evaluation during training for simplicity
    save_strategy="epoch",      # Save checkpoint at the end of each epoch
    fp16=False,                 # Enable mixed precision training for faster training and less memory usage
    bf16=False,                # Use bfloat16 if your GPU supports it (better than fp16 if supported)
    learning_rate=2e-5,        # Example learning rate
    dataloader_num_workers=0,  # Adjust based on your CPU cores and dataset size
    gradient_checkpointing=True # Enable gradient checkpointing
)

# 4. Create Trainer
print("Creating Trainer...") # Debug print
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    tokenizer=tokenizer,
    data_collator=lambda data: {
        'input_ids': torch.stack([torch.tensor(f['input_ids']) for f in data]), # Explicitly convert to tensor
        'attention_mask': torch.stack([torch.tensor(f['attention_mask']) for f in data]), # Explicitly convert to tensor
        'labels': torch.stack([torch.tensor(f['input_ids']) for f in data]) # Explicitly convert to tensor
    }
)
print("Trainer created.") # Debug print

# 5. Run Fine-tuning
print("Starting training on CPU...") # Debug print - Explicitly mention CPU
trainer.train()
print("Training complete.") # Debug print

# 6. Save Fine-tuned Model
output_model_path = "./fine_tune_slm/fine-tuned-model"
print(f"Saving model to: {output_model_path}") # Debug print
trainer.save_model(output_model_path)
print("Fine-tuning complete! Fine-tuned model saved to ./fine_tune_slm/fine-tuned-model")

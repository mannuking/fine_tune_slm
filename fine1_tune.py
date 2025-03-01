from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import Dataset
import torch
import os

# 1. Load Model and Tokenizer
model_name = "microsoft/Phi-3-mini-4k-instruct"
print(f"Loading model: {model_name}")  # Debug print

# Check for GPU availability
if torch.cuda.is_available():
    device = "cuda"
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")  # Debug print
else:
    device = "cpu"
    print("CUDA is not available. Using CPU.")  # Debug print

# Load model with mixed precision for efficiency
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"  # Auto assigns model to GPU if available
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # Set pad token to EOS token

# 2. Load and Prepare Dataset
train_file_path = "training_data.txt"
print(f"Loading dataset from: {train_file_path}")  # Debug print

with open(train_file_path, "r", encoding="utf-8") as f:
    text_data = f.read()

train_dataset = Dataset.from_dict({"text": [text_data]})  # Create dataset

# Tokenization function
def tokenize_function(examples):
    return tokenizer(
        examples["text"], 
        truncation=True, 
        padding="max_length", 
        max_length=512  # Increased max length for better training
    )

print("Tokenizing dataset...")  # Debug print
tokenized_datasets = train_dataset.map(tokenize_function, batched=True)
print("Dataset tokenized.")  # Debug print

# 3. Define Training Arguments
training_args = TrainingArguments(
    output_dir="./fine_tune_slm/results",
    num_train_epochs=3,
    per_device_train_batch_size=1,  # ✅ Reduce batch size (was 2)
    gradient_accumulation_steps=2,  # ✅ Reduce accumulation steps (was 4)
    optim="adamw_torch",
    bf16=True,  # ✅ Keep bfloat16 for better memory efficiency
    gradient_checkpointing=True,  # ✅ Helps reduce memory
    save_strategy="epoch",
    logging_steps=10,
    evaluation_strategy="no",
    report_to="none"
)

# 4. Define Data Collator (Fix for Loss Issue)
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, 
    mlm=False  # Causal LM (next-token prediction)
)

# 5. Create Trainer
print("Creating Trainer...")  # Debug print
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    tokenizer=tokenizer,
    data_collator=data_collator
)
print("Trainer created.")  # Debug print

# 6. Start Fine-tuning
print("Starting training on GPU...")  # Debug print
trainer.train()
print("Training complete.")  # Debug print

# 7. Save Fine-tuned Model
output_model_path = "./fine_tune_slm/fine-tuned-model"
print(f"Saving model to: {output_model_path}")  # Debug print
trainer.save_model(output_model_path)
print("Fine-tuning complete! Fine-tuned model saved to ./fine_tune_slm/fine-tuned-model")
import os
import json
import torch
from datasets import Dataset
from transformers import (
    GPT2LMHeadModel,
    GPT2TokenizerFast,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)

# 1. Setup paths and configuration
MODEL_PATH = "/home/selmys/LLM/test-6/final_tinystories_model"  # Update this to your local model folder
OUTPUT_DIR = "./qa_finetuned_model"
MAX_LENGTH = 256  # Hard limit based on your n_positions config

# 2. Initialize Tokenizer and Model
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2") 
tokenizer.pad_token = tokenizer.eos_token

model = GPT2LMHeadModel.from_pretrained(MODEL_PATH)
model.config.pad_token_id = tokenizer.pad_token_id

# 3. Custom Function to Load and Sanitize JSONL Data (Fixes Arrow and Number issues)
def load_and_sanitize_jsonl(filepath):
    data = {"story": [], "question": [], "answer": []}
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            # Force all inputs to string type safely
            data["story"].append(str(item.get("story", "")))
            data["question"].append(str(item.get("question", "")))
            data["answer"].append(str(item.get("answer", "")))
    return Dataset.from_dict(data)

train_dataset = load_and_sanitize_jsonl("qa_train.jsonl")
val_dataset = load_and_sanitize_jsonl("qa_val.jsonl")

# 4. Tokenization function with Prompt Loss Masking
def tokenize_qa(examples):
    concat_inputs = []
    labels = []
    
    for s, q, a in zip(examples["story"], examples["question"], examples["answer"]):
        # Structure the sequence using all three keys
        prompt = f"Story: {s.strip()}\nQuestion: {q.strip()}\nAnswer: "
        full_text = f"{prompt}{a.strip()}{tokenizer.eos_token}"
        
        tokenized_full = tokenizer(full_text, truncation=True, max_length=MAX_LENGTH)
        tokenized_prompt = tokenizer(prompt, truncation=True, max_length=MAX_LENGTH)
        
        input_ids = tokenized_full["input_ids"]
        prompt_len = len(tokenized_prompt["input_ids"])
        
        # If the story+question consumes the entire 256 window, drop the line
        if prompt_len >= MAX_LENGTH - 2:
            continue
            
        label_ids = [-100] * prompt_len + input_ids[prompt_len:]
        
        concat_inputs.append(input_ids)
        labels.append(label_ids)
        
    return {"input_ids": concat_inputs, "labels": labels}

# Map datasets individually 
tokenized_train = train_dataset.map(tokenize_qa, batched=True, remove_columns=["story", "question", "answer"])
tokenized_val = val_dataset.map(tokenize_qa, batched=True, remove_columns=["story", "question", "answer"])

# 5. Define Training Arguments (Stable for 10GB VRAM & Ampere Architecture)
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,            # Lower learning rate prevents weight explosion
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_steps=50,
    
    # Precision Settings (Bfloat16 prevents NaN errors on RTX 3080)
    fp16=False,
    bf16=torch.cuda.is_bf16_supported(),
    
    load_best_model_at_end=True,
    metric_for_best_model="loss",
    save_total_limit=2,
    report_to="none"
)

# 6. Initialize Data Collator
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    label_pad_token_id=-100,
    pad_to_multiple_of=8
)

# 7. Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    data_collator=data_collator,
)

# 8. Run Training
print("Starting fresh Supervised Fine-Tuning with Bfloat16 stability...")
trainer.train()

# 9. Save the Final Fine-Tuned Model
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Fine-tuning complete! Model saved to {OUTPUT_DIR}")


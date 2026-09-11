import math
import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
)

def main():
    # 1. Load Local Dataset Files (128GB RAM will load these instantly)
    print("Loading local train and validation text files into RAM...")
    data_files = {
        "train": "TinyStories-train.txt",
        "validation": "TinyStories-valid.txt"
    }
    raw_datasets = load_dataset("text", data_files=data_files)

    # 2. Tokenizer Setup
    # GPT-2 tokenizer natively uses <|endoftext|> as its EOS and PAD token
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 3. Efficient Sequence Chunk-Packing Setup
    context_length = 256  # 256 is the optimal sweet spot for TinyStories content

    def tokenize_function(examples):
        # Tokenize the lines of text raw
        outputs = tokenizer(examples["text"])
        
        # Concatenate all internal sequence arrays together 
        concatenated_examples = {k: sum(outputs[k], []) for k in outputs.keys()}
        total_length = len(concatenated_examples[list(outputs.keys())[0]])
        
        # Chop the stream into exact context length chunks
        if total_length >= context_length:
            total_length = (total_length // context_length) * context_length
            
        result = {
            k: [t[i : i + context_length] for i in range(0, total_length, context_length)]
            for k, t in concatenated_examples.items()
        }
        
        # For causal language models, the labels are just a copy of the input_ids
        result["labels"] = result["input_ids"].copy()
        return result

    print("Tokenizing datasets in parallel across CPU threads...")
    tokenized_train = raw_datasets["train"].map(
        tokenize_function, batched=True, remove_columns=["text"], num_proc=8
    )
    tokenized_val = raw_datasets["validation"].map(
        tokenize_function, batched=True, remove_columns=["text"], num_proc=8
    )

    # 4. Expanded Architecture (~85M Parameters) Optimized for 10GB VRAM
    print("Initializing custom ~85M parameter model architecture from scratch...")
    config = AutoConfig.from_pretrained(
        "gpt2",
        vocab_size=len(tokenizer),
        n_positions=context_length,
        n_layer=10,        # 10 Layers balances processing performance
        n_head=12,         # 12 Attention heads 
        n_embd=768,        # Hidden size matches standard GPT-2 small widths
    )
    model = AutoModelForCausalLM.from_config(config)

    # Data collator manages formatting sequences correctly for next-token prediction
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # 5. Training Arguments Tuned specifically for RTX 3080
    training_args = TrainingArguments(
        output_dir="./tinystories_llm_v2",
        # num_train_epochs=3,                 # 3 epochs provides excellent overall convergence
        max_steps = 82900,

        # RTX 3080 Native Hardware Optimization
        bf16=True,                          # Native Ampere math prevents NaN loss errors
        optim="adamw_torch",                # Robust fallback for environments without bitsandbytes
        gradient_checkpointing=True,        # Massively drops active VRAM baseline footprint
        
        # Sized perfectly for 85M parameters + 256 context lengths to use ~8.5GB VRAM
        per_device_train_batch_size=8,       # Squeezes optimal VRAM utilization
        gradient_accumulation_steps=8,       # 8 * 8 = Stable effective batch size of 64
        per_device_eval_batch_size=8,
        
        # Evaluation, Checkpointing & Tracking
        eval_strategy="steps",              
        eval_steps=500,                     # Run verification pass every 500 steps
        logging_steps=100,                  # Echo current training loss metrics
        save_steps=1000,                    # Export weight safety checkpoints
        learning_rate=4e-4,                 # Stable starting step size for an 85M scale model
        weight_decay=0.1,                   # High decay penalization limits over-memorization
        warmup_steps=1000,
        logging_dir="./logs",
        report_to="none",                  
    )

    # 6. Initialize Trainer Interface
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=data_collator,
    )

    # 7. Start Training Loop
    print("Resume pre-training run on your RTX 3080...")
    checkpoint_path = "./tinystories_llm_v2/checkpoint-46000"
    trainer.train(resume_from_checkpoint=checkpoint_path)

    # 8. Final Score Evaluation & Model Export
    print("Running final validation data evaluation pass...")
    eval_results = trainer.evaluate()
    loss = eval_results['eval_loss']
    print(f"\n======================================")
    print(f"Final Validation Loss: {loss:.4f}")
    print(f"Final Validation Perplexity: {math.exp(loss):.2f}")
    print(f"======================================\n")

    print("Saving final weights and tokenizer profiles...")
    trainer.save_model("./final_tinystories_model")
    tokenizer.save_pretrained("./final_tinystories_model")
    print("All tasks completed successfully!")

if __name__ == "__main__":
    main()


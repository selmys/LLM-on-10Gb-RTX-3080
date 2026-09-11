import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

# 1. Setup Environment
MODEL_PATH = "./qa_finetuned_model"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading stable fine-tuned model onto {DEVICE}...")
model = GPT2LMHeadModel.from_pretrained(MODEL_PATH).to(DEVICE)
tokenizer = GPT2TokenizerFast.from_pretrained(MODEL_PATH)

tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id

# 2. Test Scenario (Structured exactly like your training data schema)
test_story = (
    "Once there was a tiny robot named Sparky who lived in a messy garage. "
    "He wanted to clean the floor, so he looked for a broom. He searched behind "
    "a stack of tires and found an old, rusty toolbox instead. Inside the toolbox, "
    "he discovered a shiny silver key."
)
test_question = "What did Sparky find inside the old rusty toolbox?"

# 3. Format WITHOUT the trailing space at the end of Answer:
prompt = f"Story: {test_story.strip()}\nQuestion: {test_question.strip()}\nAnswer:"

# 4. Tokenize
inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

# 5. Run Generation (Using the stable sampling parameters)
print("\n--- Generating Answer ---")
with torch.no_grad():
    output_tokens = model.generate(
        **inputs,
        max_new_tokens=25,
        min_new_tokens=2,
        
        # --- SWITCH TO PURE GREEDY SEARCH ---
        do_sample=False,             # Turn off random sampling variations
        # Remove temperature, top_k, and top_p when do_sample=False
        
        repetition_penalty=1.1,      # Subtle penalty to prevent trailing word loops
        pad_token_id=tokenizer.eos_token_id,
        use_cache=False               
    )

# 6. Extract Raw Tokens Correctly (Avoid multi-dimensional index errors)
# Slicing the second dimension [0] from the batch, then slicing out prompt length
raw_input_len = inputs["input_ids"].shape[1]
generated_tokens_tensor = output_tokens[0][raw_input_len:]
generated_tokens_list = generated_tokens_tensor.tolist()

print(f"PROMPT STRUCTURE:\n{prompt}")
print(f"Raw Token IDs Generated: {generated_tokens_list}")

# 7. Decode Outputs
verbose_answer = tokenizer.decode(generated_tokens_list, skip_special_tokens=False)
print(f"Verbose Decoded Output: [{verbose_answer}]")

clean_answer = tokenizer.decode(generated_tokens_list, skip_special_tokens=True)
print(f"Standard Decoded Output: [{clean_answer.strip()}]")


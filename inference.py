import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 1. Path to your finished model weights
model_path = "./final_tinystories_model"

print("Loading your custom TinyStories model...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)
params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Model Parameters: ", params)

# Move the model to your RTX 3080 to keep generation fast
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# 2. Craft a creative prompt fitting the TinyStories style
prompt = "Tommy was playing with his red ball in the park."

# 3. FIX: Generate input IDs AND the explicit attention mask
inputs = tokenizer(prompt, return_tensors="pt")
input_ids = inputs["input_ids"].to(device)
attention_mask = inputs["attention_mask"].to(device)

print(f"\n--- Generating story based on prompt: ---\n'{prompt}'\n")

# 4. Configure generation settings
output_tokens = model.generate(
    input_ids=input_ids,
    attention_mask=attention_mask,   # FIX: Pass the attention mask here
    max_length=200,                  # Maximum story length
    do_sample=True,                  # Enable creative sampling
    temperature=0.7,                 # Balanced creativity setting
    top_p=0.9,                       # Nucleus sampling
    repetition_penalty=1.2,          # Discourages sentence looping
    pad_token_id=tokenizer.eos_token_id
)

# 5. Decode tokens back into human-readable text
generated_story = tokenizer.decode(output_tokens[0], skip_special_tokens=True)

# 6. CLEANUP TRICK: Find the very last sentence-ending punctuation mark
last_punctuation = max(
    generated_story.rfind("."), 
    generated_story.rfind("!"), 
    generated_story.rfind("?")
)
if last_punctuation != -1:
    # Cut off any trailing partial sentences so the story ends cleanly
    generated_story = generated_story[:last_punctuation + 1]

print("--- Generated Output: ---")
print(generated_story)
print("-------------------------")


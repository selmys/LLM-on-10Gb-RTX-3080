import re
import json
import jsonlines
from vllm import LLM, SamplingParams

# ==========================================
# CONFIGURATION SETTINGS
# ==========================================
MODEL_NAME = "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4"
MAX_STORIES_TO_PROCESS = 15000                  # Limits execution count for test runs
OUTPUT_FILE = "train_qa_dataset.jsonl"

print("Initializing vLLM Engine...")
# gpu_memory_utilization=0.8 leaves 2GB headroom on your 10GB card for desktop/OS tasks
llm = LLM(model=MODEL_NAME, gpu_memory_utilization=0.8, max_model_len=4096)

# Set generation constraints: high temperature allows creative question framing
sampling_params = SamplingParams(temperature=0.7, max_tokens=300)

# ==========================================
# 1. READ AND EXTRACT TINYSTORIES CLEANLY
# ==========================================
print("Reading tinystories.txt...")
with open("TinyStories-train.txt", "r", encoding="utf-8") as f:
    raw_content = f.read()

print("Parsing text into individual stories via delimiters...")
# Split by standard HuggingFace dataset tags <|endoftext|> OR double-newlines
#raw_stories = re.split(r"<\|endoftext\|>|\n\s*\n", raw_content)
raw_stories = re.split(r"<|endoftext|>", raw_content)

# Clean up stories and discard empty lines or giant corrupted clusters
selected_stories = []
for story in raw_stories:
    cleaned = story.strip()
    # Ensure the story is long enough to be meaningful, but short enough to fit context
    if 100 < len(cleaned) < 4000:
        selected_stories.append(cleaned)
        if len(selected_stories) >= MAX_STORIES_TO_PROCESS:
            break

print(f"Successfully extracted {len(selected_stories)} individual stories.")

# ==========================================
# 2. CONSTRUCT LLM BATCH CHAT PROMPTS
# ==========================================
batch_chats = []

system_instruction = (
    "You are an expert AI dataset generator. Read the provided short children's story. "
    "Generate exactly 1 specific factual question about an event or entity in the story, "
    "and provide its short direct answer. "
    "You MUST respond in raw JSON format with exactly two keys: 'question' and 'answer'. "
    "Do not include markdown blocks like ```json."
)

for story in selected_stories:
    # Explicitly verification check: Make sure ONLY the single story goes into this loop iteration!
    user_prompt = f"Story:\n{story}\n\nGenerate the JSON question and answer pair based on this text."
    
    chat_history = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": user_prompt}
    ]
    batch_chats.append(chat_history)

# ==========================================
# 3. EXECUTE BATCH GENERATION (vLLM Parallelism)
# ==========================================
print(f"Feeding {len(batch_chats)} stories into vLLM in parallel. Processing...")
# vLLM automatically formats via the model's native Jinja template during chat call
outputs = llm.chat(messages=batch_chats, sampling_params=sampling_params)

# ==========================================
# 4. PARSE OUTPUTS AND SAVE TO JSONL
# ==========================================
print(f"Generation complete! Writing structured results to {OUTPUT_FILE}...")

with jsonlines.open(OUTPUT_FILE, mode='w') as writer:
    for i, output in enumerate(outputs):
        raw_text = output.outputs[0].text.strip()
        story_context = selected_stories[i]
        
        try:
            # Clean up accidental markdown syntax wrappers if the model generated them
            clean_json_str = re.sub(r"^```json\s*|\s*```$", "", raw_text, flags=re.MULTILINE).strip()
            parsed_qa = json.loads(clean_json_str)
            
            # Combine the original text back into the record structure
            dataset_record = {
                "story": story_context,
                "question": parsed_qa["question"],
                "answer": parsed_qa["answer"]
            }
            
            # Write a single line of raw JSON to disk immediately
            writer.write(dataset_record)
            
        except Exception as e:
            # Skip invalid outputs without crashing the batch run
            print(f"Skipping index {i} due to JSON malformation: {e}")
            print(f"Raw Model Text was: {raw_text}\n")

print(f"Done! Dataset ready for instruction fine-tuning.")

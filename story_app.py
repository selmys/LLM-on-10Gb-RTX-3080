import gradio as gr
import torch
import re
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# 1. System Setup
device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = "/home/selmys/LLM/test-6/final_tinystories_model"

print(f"Loading custom TinyStories model onto {device} for Web UI...")

# 2. Load Model & Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path).to(device)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 3. Story Generation and Post-Processing Logic
def generate_story(prompt_text, max_new_tokens, temperature, top_p, repetition_penalty):
    if not prompt_text.strip():
        return "Please enter a story starter prompt first!"
        
    # Format the input tensor
    inputs = tokenizer(prompt_text, return_tensors="pt", padding=True).to(device)
    
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=int(max_new_tokens),
            temperature=float(temperature),
            top_p=float(top_p),
            repetition_penalty=float(repetition_penalty),
            do_sample=True if temperature > 0.0 else False,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Extract only the generated tokens (skipping the prompt batch dimension)
    input_length = inputs.input_ids.shape[1]
    generated_tokens = output_ids[0][input_length:]
    story_output = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    
    # Format: Re-attach prompt text to the output for a seamless read
    full_narrative = f"{prompt_text.strip()} {story_output}"
    
    # Post-Processing: Clean up any trailing broken sentences at the very end
    sentence_endings = [m.start() for m in re.finditer(r'[.!?]', full_narrative)]
    if sentence_endings:
        last_punctuation_index = sentence_endings[-1]
        full_narrative = full_narrative[:last_punctuation_index + 1]
    else:
        if len(full_narrative) > len(prompt_text):
            full_narrative += "..."
            
    return full_narrative

# 4. Build the Web Interface Layout
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 📖 TinyStories Creative Writer UI")
    gr.Markdown("Type a starting line below to let your custom 110M parameter model generate a full children's story.")
    
    with gr.Row():
        # Left Side: Controls & Input
        with gr.Column(scale=1):
            prompt_input = gr.Textbox(
                label="Story Starter / Prompt", 
                lines=3, 
                placeholder="e.g., Tommy was playing with his red ball in the park.",
                value="Tommy was playing with his red ball in the park."
            )
            
            with gr.Accordion("Story Controls", open=True):
                max_tokens = gr.Slider(minimum=20, maximum=250, value=150, step=5, label="Story Length (Max Tokens)")
                temp = gr.Slider(minimum=0.1, maximum=1.5, value=0.7, step=0.1, label="Creativity (Temperature)")
                top_p_slider = gr.Slider(minimum=0.1, maximum=1.0, value=0.9, step=0.05, label="Word Variety (Top-p)")
                rep_penalty = gr.Slider(minimum=1.0, maximum=2.0, value=1.1, step=0.05, label="Repetition Penalty")
            
            generate_btn = gr.Button("Generate Story ✨", variant="primary")
            
        # Right Side: Final Narrative Output
        with gr.Column(scale=1):
            story_output_box = gr.Textbox(
                label="Generated Children's Story", 
                lines=12, 
                interactive=False
            )
            
    # Connect Layout Actions to Python Functions
    generate_btn.click(
        fn=generate_story,
        inputs=[prompt_input, max_tokens, temp, top_p_slider, rep_penalty],
        outputs=story_output_box
    )

# 5. Launch the local web application server
if __name__ == "__main__":
    demo.launch()


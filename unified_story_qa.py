import gradio as gr
import torch
import re
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# 1. Configuration & Setup
device = "cuda" if torch.cuda.is_available() else "cpu"
story_model_path = "/home/selmys/LLM/test-6/final_tinystories_model"
qa_model_path = "/home/selmys/LLM/test-6/qa_finetuned_model"

print(f"Initializing Unified App on {device}...")

# 2. Load Models and Tokenizers
print("-> Loading Storyteller Model...")
story_tokenizer = GPT2Tokenizer.from_pretrained(story_model_path)
story_model = GPT2LMHeadModel.from_pretrained(story_model_path).to(device)
if story_tokenizer.pad_token is None:
    story_tokenizer.pad_token = story_tokenizer.eos_token

print("-> Loading Q&A Fine-tuned Model...")
qa_tokenizer = GPT2Tokenizer.from_pretrained(qa_model_path)
qa_model = GPT2LMHeadModel.from_pretrained(qa_model_path).to(device)
if qa_tokenizer.pad_token is None:
    qa_tokenizer.pad_token = qa_tokenizer.eos_token


# 3. Step 1: Story Generation Logic
def generate_story(prompt_text, max_new_tokens, temperature, top_p):
    if not prompt_text.strip():
        return "Please enter a story starter prompt first!"
        
    inputs = story_tokenizer(prompt_text, return_tensors="pt", padding=True).to(device)
    input_length = inputs.input_ids.shape[1]
    
    with torch.no_grad():
        output_ids = story_model.generate(
            **inputs,
            max_new_tokens=int(max_new_tokens),
            temperature=float(temperature),
            top_p=float(top_p),
            do_sample=True if temperature > 0.0 else False,
            pad_token_id=story_tokenizer.eos_token_id
        )
    
    generated_tokens = output_ids[0][input_length:]
    story_output = story_tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    full_narrative = f"{prompt_text.strip()} {story_output}"

    # Case-insensitive cleanup of repeating "The end" phrases
    full_narrative = re.sub(r'(?i)(\s*the\s+end\.?)+', ' The end.', full_narrative)

    # Post-Processing: Clean up any trailing broken sentences
    sentence_endings = [m.start() for m in re.finditer(r'[.!?]', full_narrative)]
    if sentence_endings:
        full_narrative = full_narrative[:sentence_endings[-1] + 1]
    
    # Return the story, clear out any old answers, and unlock the Q&A panel
    return full_narrative, "", gr.update(interactive=True), gr.update(interactive=True)


# 4. Step 2: Contextual Q&A Logic
def answer_question(story_context, question_text):
    if not story_context.strip():
        return "No story context found. Please generate a story first!"
    if not question_text.strip():
        return "Please enter a question!"
    
    # Format the prompt using your exact fine-tuning structure
    qa_prompt = f"Context: {story_context.strip()}\nQuestion: {question_text.strip()}\nAnswer:"
    
    inputs = qa_tokenizer(qa_prompt, return_tensors="pt", padding=True).to(device)
    input_length = inputs.input_ids.shape[1] # Use shape[1] to get token count dimension
    
    with torch.no_grad():
        output_ids = qa_model.generate(
            **inputs,
            max_new_tokens=50,  # <-- FIX: Changed from int(max_new_tokens) to a fixed integer
            temperature=0.2,    # Low temperature to keep answers grounded in facts
            do_sample=True,
            pad_token_id=qa_tokenizer.eos_token_id,
            no_repeat_ngram_size=3 # Prevents the QA model from looping answers
        )
        
    generated_tokens = output_ids[0][input_length:]
    answer = qa_tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    return answer

# 5. Interface Layout (Two-Step Workflow)
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 📚 TinyStories: Interactive Narrative & QA Engine")
    gr.Markdown("Step 1: Spark a new story. Step 2: Interrogate the fine-tuned model about what happened.")
    
    # --- PHASE 1: STORY GENERATION ---
    gr.Markdown("### 📜 Step 1: Generate the Story")
    with gr.Row():
        with gr.Column(scale=1):
            prompt_input = gr.Textbox(
                label="Story Starter / Prompt", 
                lines=3, 
                value="Tommy was playing with his red ball in the park."
            )
            with gr.Accordion("Story Controls", open=False):
                max_tokens = gr.Slider(minimum=50, maximum=250, value=150, step=5, label="Max New Tokens")
                temp = gr.Slider(minimum=0.1, maximum=1.5, value=0.7, step=0.1, label="Temperature")
                top_p_slider = gr.Slider(minimum=0.1, maximum=1.0, value=0.9, step=0.05, label="Top-p")
            
            generate_btn = gr.Button("Generate Narrative ✨", variant="primary")
            
        with gr.Column(scale=1):
            # This output text block serves as the context window for Phase 2
            story_output_box = gr.Textbox(
                label="Generated Narrative (Acts as Context for Questions)", 
                lines=10, 
                interactive=False
            )
            
    gr.Markdown("---")
    
    # --- PHASE 2: CONTEXTUAL Q&A ---
    gr.Markdown("### ❓ Step 2: Ask Questions About the Story")
    with gr.Row():
        with gr.Column(scale=1):
            question_input = gr.Textbox(
                label="Your Question", 
                placeholder="e.g., What color was Tommy's ball?",
                interactive=False # Locked until a story is generated
            )
            ask_btn = gr.Button("Ask Model 🤔", variant="secondary", interactive=False)
            
        with gr.Column(scale=1):
            answer_output_box = gr.Textbox(
                label="Model Answer", 
                lines=4, 
                interactive=False
            )

    # --- UI EVENT LINKING ---
    # Trigger story generation
    generate_btn.click(
        fn=generate_story,
        inputs=[prompt_input, max_tokens, temp, top_p_slider],
        outputs=[story_output_box, answer_output_box, question_input, ask_btn]
    )
    
    # Trigger question answering
    ask_btn.click(
        fn=answer_question,
        inputs=[story_output_box, question_input],
        outputs=answer_output_box
    )

if __name__ == "__main__":
    demo.launch(share=True)


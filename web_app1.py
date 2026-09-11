import gradio as gr
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# 1. Setup device and load your model
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading fine-tuned model onto {device} for Web UI...")

model_path = "./final_tinystories_model"  # Update this to your model's actual path
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path).to(device)

# Ensure pad token is set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 2. Updated generation function accepting slider inputs
def generate_answer(story, question, max_new_tokens, temperature, top_p, repetition_penalty):
    # Format the prompt exactly how the model was trained
    prompt = f"Story: {story}\nQuestion: {question}\nAnswer:"
    
    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(device)
    
    # Generate text using the dynamic slider values
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
    
    # Decode only the newly generated tokens
    input_length = inputs.input_ids.shape[1]
    generated_tokens = output_ids[0][input_length:]
    answer = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    return answer.strip()

# 3. Build the Gradio Interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🧠 My DIY Tiny LLM - Story QA Interface")
    
    with gr.Row():
        # Left column: Inputs and Settings Sidebar
        with gr.Column(scale=2):
            story_input = gr.Textbox(label="Context / Story", lines=6, placeholder="Paste your short story here...")
            question_input = gr.Textbox(label="Question", lines=2, placeholder="What would you like to ask?")
            
            # Sidebar container for sliding configurations
            with gr.Accordion("Generation Settings", open=True):
                max_tokens = gr.Slider(minimum=10, maximum=100, value=40, step=1, label="Max New Tokens")
                temp = gr.Slider(minimum=0.0, maximum=1.5, value=0.7, step=0.1, label="Temperature (Creativity)")
                top_p_slider = gr.Slider(minimum=0.1, maximum=1.0, value=0.9, step=0.05, label="Top-p (Nucleus Sampling)")
                rep_penalty = gr.Slider(minimum=1.0, maximum=2.0, value=1.1, step=0.05, label="Repetition Penalty")
            
            submit_btn = gr.Button("Generate Answer", variant="primary")
            
        # Right column: Output Window
        with gr.Column(scale=1):
            answer_output = gr.Textbox(label="Predicted Answer", lines=4, interactive=False)
            
    # Connect UI components to the Python function
    submit_btn.click(
        fn=generate_answer,
        inputs=[story_input, question_input, max_tokens, temp, top_p_slider, rep_penalty],
        outputs=answer_output
    )

# 4. Launch web server (Passing theme fix to launch() for Gradio 6+)
if __name__ == "__main__":
    demo.launch()


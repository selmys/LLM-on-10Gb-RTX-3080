import re
import torch
import gradio as gr
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

# 1. Environment & Model Setup
MODEL_PATH = "./qa_finetuned_model"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading fine-tuned model onto {DEVICE} for Web UI...")
model = GPT2LMHeadModel.from_pretrained(MODEL_PATH).to(DEVICE)
tokenizer = GPT2TokenizerFast.from_pretrained(MODEL_PATH)

tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id

# 2. Prediction Function
def generate_answer(story, question, max_new_tokens, temperature, top_p, repetition_penalty):
    prompt = f"Story: {story}\nQuestion: {question}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(DEVICE)
    
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
    
    # Extract only the newly generated tokens
    input_length = inputs.input_ids.shape[1]
    generated_tokens = output_ids[0][input_length:]
    answer = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    
    # --- Post-Processing: Clean up interrupted sentences ---
    # Find the last occurrence of any sentence-ending punctuation (. ! ?)
    sentence_endings = [m.start() for m in re.finditer(r'[.!?]', answer)]
    
    if sentence_endings:
        # Keep everything up to and including the last complete sentence punctuation
        last_punctuation_index = sentence_endings[-1]
        answer = answer[:last_punctuation_index + 1]
    else:
        # Fallback: If the model didn't even complete one sentence, add an ellipsis
        if len(answer) > 0:
            answer += "..."
            
    return answer

# 3. Build the Interface Layout
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🤖 Tiny Local DIY LLM — QA Interface")
    gr.Markdown(
        "Provide a background story context and a question. "
        "The model will use its fine-tuned extraction layer to pinpoint the answer."
    )
    
    with gr.Row():
        with gr.Column(scale=2):
            story_input = gr.Textbox(
                label="Story Context", 
                placeholder="Once there was a tiny robot named Sparky...", 
                lines=8
            )
            question_input = gr.Textbox(
                label="Question", 
                placeholder="What did Sparky find?", 
                lines=2
            )
            submit_btn = gr.Button("Generate Answer", variant="primary")
            
        with gr.Column(scale=1):
            answer_output = gr.Textbox(
                label="Model Response", 
                interactive=False, 
                lines=6
            )
            
            # Helpful quick examples to click and load instantly
            gr.Examples(
                examples=[
                    [
                        "Once there was a tiny robot named Sparky who lived in a messy garage. He wanted to clean the floor, so he looked for a broom. He searched behind a stack of tires and found an old, rusty toolbox instead. Inside the toolbox, he discovered a shiny silver key.",
                        "What did Sparky find inside the old rusty toolbox?"
                    ],
                    [
                        "The little boy worked hard and completed all the tasks. He brought the old man the apples, the river stone, and the feather. The old man was very impressed and told him the answer to the question.",
                        "What objects did the boy bring to the old man?"
                    ]
                ],
                inputs=[story_input, question_input]
            )
            
    # Connect UI button click event to logic
    submit_btn.click(
        fn=generate_answer, 
        inputs=[story_input, question_input], 
        outputs=answer_output
    )

# 4. Launch Local Web Server
if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)


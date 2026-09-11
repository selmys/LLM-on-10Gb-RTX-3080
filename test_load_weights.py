import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

try:
    print("Testing if your fine-tuned model loads properly...")
    # Update to point to your saved directory
    model = GPT2LMHeadModel.from_pretrained("./qa_finetuned_model")
    tokenizer = GPT2TokenizerFast.from_pretrained("./qa_finetuned_model")
    print("🎉 Success! The saved model is structurally intact.")
except Exception as e:
    print(f"❌ Error: Your saved model folder is corrupted due to the NaN loss loop: {e}")

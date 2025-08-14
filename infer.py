import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from prompt_templates import format_example

BASE_MODEL = os.environ.get("BASE_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
ADAPTER_DIR = os.environ.get("ADAPTER_DIR", "outputs/tinyllama-domain-lora-cpu")

def load_model():
    tok = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
    model = PeftModel.from_pretrained(base, ADAPTER_DIR)
    model = model.to("cpu")
    model.eval()
    return tok, model

def generate(domain, instruction, input_text, max_new_tokens=256, temperature=0.8, top_p=0.9):
    tok, model = load_model()
    prompt = format_example(domain, instruction, input_text)
    ids = tok(prompt, return_tensors="pt")
    out = model.generate(
        **ids,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=tok.eos_token_id,
    )
    text = tok.decode(out[0], skip_special_tokens=True)
    return text.split("[RESPONSE]")[-1].strip()

if __name__ == "__main__":
    print(generate(
        domain="marketing",
        instruction="Write a LinkedIn post announcing our AI-powered editor.",
        input_text="Brand: TypoZero; Audience: marketers; CTA: Try free today"
    ))

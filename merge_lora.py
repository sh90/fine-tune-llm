# merge_lora.py
from transformers import AutoModelForCausalLM
from peft import PeftModel
import torch, os

BASE="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
ADAPTER="outputs/tinyllama-domain-lora"
OUT="outputs/tinyllama-domain-merged"

base = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.float16, device_map="auto")
peft = PeftModel.from_pretrained(base, ADAPTER)
merged = peft.merge_and_unload()
merged.save_pretrained(OUT)

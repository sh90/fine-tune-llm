Use TinyLlama-1.1B-Chat (fast, permissive) for the live fine-tune. If you’ve got a decent GPU (≥8GB VRAM), 
you can swap to Qwen2.5-1.5B-Instruct. 
Mistral/Llama 7B will be too slow for a 2-hour session on most laptops.

## Steps:

Build a small domain dataset (marketing, legal, tech docs)

QLoRA fine-tune with PEFT (LoRA rank 8)

Prompt template + inference

Quick sanity eval 



## Topics
What are open source LLMs? - https://huggingface.co/

LoRA/PEFT: we’re training a small set of low-rank matrices; fast and cheap.

QLoRA: quantize base model to 4-bit, keep optimizer in 16-bit, preserve quality.

Prompt templating: consistent structure = stabler generations across domains.

Data curation > data volume: even 50–200 good pairs can shift tone & structure.

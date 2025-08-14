Use TinyLlama-1.1B-Chat (fast, permissive) for the live fine-tune. If you’ve got a decent GPU (≥8GB VRAM), 
you can swap to Qwen2.5-1.5B-Instruct. 
Mistral/Llama 7B will be too slow for a 2-hour session on most laptops.

## Steps:

Build a small domain dataset (marketing, legal, tech docs)

QLoRA fine-tune with PEFT (LoRA rank 8)

Prompt template + inference

Quick sanity eval (few-shot prompts)

(Optional) Merge LoRA into base weights for export

## Agenda
0–10 min – Setup & explain LoRA/QLoRA + dataset format
10–25 min – Create a tiny domain dataset (JSONL)
25–75 min – Fine-tune with TRL + PEFT (1 epoch)
75–95 min – Inference with a clean prompt template
95–110 min – Quick eval on held-out prompts (3 domains)
110–120 min – (Optional) Merge adapters & model card notes


## Topics
LoRA/PEFT: we’re training a small set of low-rank matrices; fast and cheap.

QLoRA: quantize base model to 4-bit, keep optimizer in 16-bit, preserve quality.

Prompt templating: consistent structure = stabler generations across domains.

Data curation > data volume: even 50–200 good pairs can shift tone & structure.
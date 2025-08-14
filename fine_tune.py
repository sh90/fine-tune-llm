import json, os
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, TaskType
from trl import SFTTrainer, SFTConfig
from prompt_templates import format_example

BASE_MODEL = os.environ.get("BASE_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "outputs/tinyllama-domain-lora")

def tokenize_fn(ex, tokenizer):
    text = format_example(ex["domain"], ex["instruction"], ex.get("input",""))
    # we train the model to generate only the output; include the prompt as context
    full = text + (ex["output"] if ex.get("output") else "")
    return tokenizer(full, truncation=True, max_length=1024)

def main():
    # Dataset
    ds = load_dataset("json", data_files={"train":"data/train.jsonl","eval":"data/eval.jsonl"})
    # Tokenizer
    tok = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # 4-bit quantization (QLoRA)
    qconfig = BitsAndBytesConfig(load_in_4bit=True,
                                 bnb_4bit_use_double_quant=True,
                                 bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype="bfloat16")

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=qconfig,
        device_map="auto"
    )

    # LoRA config (small for laptops)
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj","v_proj","k_proj","o_proj","gate_proj","up_proj","down_proj"]  # safe default; model-aware
    )

    # Tokenize
    tokenized = ds.map(lambda ex: tokenize_fn(ex, tok), remove_columns=ds["train"].column_names)

    train_cfg = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=True,
        logging_steps=5,
        save_steps=50,
        max_grad_norm=0.3,
        packing=True
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tok,
        peft_config=lora_cfg,
        args=train_cfg,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["eval"]
    )

    trainer.train()
    trainer.save_model()
    tok.save_pretrained(OUTPUT_DIR)

if __name__ == "__main__":
    main()

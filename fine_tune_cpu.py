import os
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, TaskType
from prompt_templates import format_example

BASE_MODEL = os.environ.get("BASE_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "outputs/tinyllama-domain-lora-cpu")
MAX_LENGTH = int(os.environ.get("MAX_LENGTH", 512))
SEED = int(os.environ.get("SEED", 42))

def build_text(example):
    # Combine prompt + target response
    return format_example(
        example["domain"],
        example["instruction"],
        example.get("input", "")
    ) + (example.get("output") or "")

def tokenize_only(ex, tokenizer, max_len):
    # No labels here; collator will create labels and pad batches uniformly
    return tokenizer(
        build_text(ex),
        truncation=True,
        max_length=max_len,
        padding=False
    )

def main():
    # 1) Data
    ds = load_dataset("json", data_files={"train": "data/train.jsonl", "eval": "data/eval.jsonl"})

    # 2) Tokenizer
    tok = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    # 3) Tokenize (no labels here)
    cols = ds["train"].column_names
    tokenized = ds.map(lambda ex: tokenize_only(ex, tok, MAX_LENGTH), remove_columns=cols)

    # 4) Base model (CPU/MPS) + LoRA
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)

    # Gradient checkpointing & cache setting (safe on CPU/MPS; helps memory)
    try:
        model.gradient_checkpointing_enable()
    except Exception:
        pass
    # Needed when gradient checkpointing is on
    try:
        model.config.use_cache = False
    except Exception:
        pass

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    )
    model = get_peft_model(model, lora_cfg)

    # 5) Collator: causal LM (creates labels + pads batches)
    collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False)

    # 6) Training args — minimal & old-version-friendly
    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        seed=SEED,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,   # effective batch size = 4
        learning_rate=2e-4,
        logging_steps=10,
        save_steps=200,
        dataloader_pin_memory=False,     # silences MPS pin_memory warning
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["eval"],  # used in manual eval below
        tokenizer=tok,
        data_collator=collator,
    )

    # 7) Train
    trainer.train()
    trainer.save_model()
    tok.save_pretrained(OUTPUT_DIR)

    # 8) Manual eval (now padded properly)
    try:
        metrics = trainer.evaluate(eval_dataset=tokenized["eval"])
        print("Eval metrics:", metrics)
    except Exception as e:
        print("Eval skipped:", repr(e))

if __name__ == "__main__":
    main()

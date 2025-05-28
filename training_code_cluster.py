import torch
import os
import json
from datasets import Dataset, load_from_disk
from swift import (
    Swift,
    LoraConfig,
    Trainer,
    TrainingArguments
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    set_seed,
    default_data_collator
)
from torch.distributed import init_process_group, destroy_process_group

MODEL_DIR = "./qwen2.5-32B"
DATASET_JSON_DIR_PATH = "dataset"
MODEL_OUTPUT_DIR = "swift_multi_gpu_checkpoints" # the code saves checkpoints successfully, but error occurs at the end of training when merging. You can easily bring any saved lora checkpoints and merge it with base model.
###DEBUGGING INFO
print(f"[RANK {os.environ.get('RANK')}] Running on node {os.uname().nodename}, GPU count: {torch.cuda.device_count()}, visible devices: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
##########

# Set seed for reproducibility
set_seed(42)

def ddp_setup():
    init_process_group("nccl", init_method="env://")
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    print(f"[ddp_setup] local_rank={local_rank}, rank={rank}, world_size={world_size}")
    return local_rank, rank, world_size

def load_and_tokenize_dataset(data_dir, tokenizer, max_length=1024):
    cache_dir = "toked_dataset"
    if os.path.exists(cache_dir):
        print("Loading tokenized dataset from cache...")
        return load_from_disk(cache_dir)

    print(f"Loading data from {data_dir}")
    all_texts = []

    for root, _, files in os.walk(data_dir):
        for filename in files:
            if filename.endswith('.json'):
                path = os.path.join(root, filename)
                try:
                    with open(path, "r", encoding="utf-8") as fp:
                        data = json.load(fp)
                        if isinstance(data, dict) and "conversation" in data:
                            convo = data["conversation"]
                            messages = []
                            for msg in convo:
                                role = msg.get("role", "")
                                text = msg.get("text", "").strip()
                                if role and text:
                                    if role == "user":
                                        messages.append(f"<|user|>\n{text}")
                                    elif role == "assistant":
                                        messages.append(f"<|assistant|>\n{text}")
                            full_convo = "\n".join(messages)
                            all_texts.append(full_convo)
                except Exception as e:
                    print(f"Skipping {path}: {e}")

    print(f"Total conversations loaded: {len(all_texts)}")

    dataset = Dataset.from_dict({
        "text": all_texts
    })

    def tokenize(examples):
        tokenized = tokenizer(
            examples["text"],
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_tensors=None
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    tokenized_dataset = dataset.map(
        tokenize,
        batched=True,
        num_proc=4,
        remove_columns=["text"],
        desc="Tokenizing dataset",
    )

    tokenized_dataset.save_to_disk(cache_dir)
    return tokenized_dataset

def main():
    local_rank, rank, world_size = ddp_setup()
    torch.cuda.set_device(local_rank)

    model_path = MODEL_DIR
    data_dir = DATASET_JSON_DIR_PATH
    output_dir = MODEL_OUTPUT_DIR
    merged_model_path = os.path.join(output_dir, "final_merged")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=True,
        padding_side="right",
    )
    tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        inference_mode=False
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    model = Swift.prepare_model(model, lora_config)
    model.enable_input_require_grads()
    model.config.use_cache = False

    if local_rank == 0:
        print("Loading dataset...")

    train_dataset = load_and_tokenize_dataset(data_dir, tokenizer)

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=1,
        learning_rate=1e-5,
        weight_decay=0.01,
        bf16=True,
        logging_steps=10,
        save_steps=500,
        dataloader_num_workers=2,
        dataloader_pin_memory=True,
        dataloader_drop_last=True,
        remove_unused_columns=False,
        optim="adamw_torch",
        save_strategy="steps",
        load_best_model_at_end=False,
        ddp_find_unused_parameters=False,
        ddp_backend="nccl",
        local_rank=local_rank,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
    )

    if local_rank == 0:
        print("\nStarting training...")

    trainer.train()

    if local_rank == 0:
        print("\nMerging LoRA weights with base model...")
        merged_model = Swift.merge_and_unload(model)
        merged_model.save_pretrained(merged_model_path, safe_serialization=True)
        tokenizer.save_pretrained(merged_model_path)

    destroy_process_group()

if __name__ == "__main__":
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["NCCL_P2P_DISABLE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main()

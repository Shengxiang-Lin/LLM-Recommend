from datasets import load_dataset, DatasetDict
import random
from multiprocessing import cpu_count
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
import torch
import os
from trl import SFTTrainer
from peft import LoraConfig

#Load datasets
raw_datasets = load_dataset("./ultrachat_200k")
indices = range(0, 1000)
dataset_dict = {
    "train": raw_datasets["train_sft"].select(indices),
    "test": raw_datasets["test_sft"].select(indices)
}
raw_datasets = DatasetDict(dataset_dict)
# Configuring the tokenizer
local_model_path = "./mistral-7b"
tokenizer = AutoTokenizer.from_pretrained(local_model_path)
tokenizer.pad_token = tokenizer.eos_token  # Set pad_token explicitly
tokenizer.model_max_length = 2048  # Specify the maximum length
# Configuring the chat template
DEFAULT_CHAT_TEMPLATE = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"
tokenizer.chat_template = DEFAULT_CHAT_TEMPLATE
# Processing datasets
def apply_chat_template(example, tokenizer):
    messages = example["messages"]
    if messages[0]["role"] != "system":
        messages.insert(0, {"role": "system", "content": ""})
    example["text"] = tokenizer.apply_chat_template(messages, tokenize=False)
    return example

column_names = list(raw_datasets["train"].features)
raw_datasets = raw_datasets.map(
    apply_chat_template,
    num_proc=cpu_count(),
    fn_kwargs={"tokenizer": tokenizer},
    remove_columns=column_names,
    desc="Applying chat template",
)
train_dataset = raw_datasets["train"]
eval_dataset = raw_datasets["test"]
# 4bit quantized configuration
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="bfloat16",  # Compatible with 4bit quantization
)
# load model (no device_map specified, managed by DeepSpeed)
model = AutoModelForCausalLM.from_pretrained(
    local_model_path,
    attn_implementation="flash_attention_2" if torch.cuda.is_available() else "eager",
    torch_dtype="auto",
    use_cache=False,  # Disable caching, compatible with gradient checkpoints
    quantization_config=quantization_config,
)
output_dir = 'data/zephyr-7b-sft-lora'
training_args = TrainingArguments(
    fp16=True, # specify bf16=True instead when training on GPUs that support bf16
    do_eval=True,
    eval_strategy="epoch",
    gradient_accumulation_steps=32,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    learning_rate=2.0e-05,
    log_level="info",
    logging_steps=1,
    logging_strategy="steps",
    lr_scheduler_type="cosine",
    max_steps=-1,
    num_train_epochs=3,
    output_dir=output_dir,
    overwrite_output_dir=True,
    per_device_eval_batch_size=8, # originally set to 8
    per_device_train_batch_size=8, # originally set to 8
    # push_to_hub=True,
    # hub_model_id="zephyr-7b-sft-lora",
    # hub_strategy="every_save",
    # report_to="tensorboard",
    save_strategy="no",
    save_total_limit=None,
    seed=42,
)
# based on config
peft_config = LoraConfig(
        r=16,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)
trainer = SFTTrainer(
    model=model, 
    args=training_args, 
    train_dataset=train_dataset,  
    eval_dataset=eval_dataset,  
    processing_class=tokenizer,  
    formatting_func=lambda x: x["text"],  
    peft_config=peft_config, 
)
# Initiate training
train_result = trainer.train()
# Save Model
metrics = train_result.metrics
metrics["train_samples"] = len(train_dataset)
trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)
trainer.save_state()
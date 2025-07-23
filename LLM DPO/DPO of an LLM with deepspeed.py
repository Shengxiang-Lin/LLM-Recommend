from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM, TrainingArguments
import re
from multiprocessing import cpu_count
import random
from peft import PeftConfig, PeftModel, LoraConfig
import torch
import os
from trl import DPOTrainer

#Load datasets
raw_datasets = load_dataset("./ultrafeedback_binarized")
indices = range(0,1000)
dataset_dict = {"train": raw_datasets["train_prefs"].select(indices),
                "test": raw_datasets["test_prefs"].select(indices)}
raw_datasets = DatasetDict(dataset_dict)
example = raw_datasets["train"][0]
print(example.keys())
# Configuring the tokenizer
model_id = "alignment-handbook/zephyr-7b-sft-lora"
tokenizer = AutoTokenizer.from_pretrained("./mistral-7b")
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
# Truncate from left to ensure we don't lose labels in final turn
tokenizer.truncation_side = "left"
# Set reasonable default for models without max length
if tokenizer.model_max_length > 100_000:
    tokenizer.model_max_length = 2048
DEFAULT_CHAT_TEMPLATE = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"
tokenizer.chat_template = DEFAULT_CHAT_TEMPLATE
# Configuring the chat template
def apply_chat_template(example, tokenizer, assistant_prefix="<|assistant|>\n"):
    def _strip_prefix(s, pattern):
        # Use re.escape to escape any special characters in the pattern
        return re.sub(f"^{re.escape(pattern)}", "", s)
    if all(k in example.keys() for k in ("chosen", "rejected")):
            # Compared to reward modeling, we filter out the prompt, so the text is everything after the last assistant token
            prompt_messages = [[msg for msg in example["chosen"] if msg["role"] == "user"][0]]
            # Insert system message
            if example["chosen"][0]["role"] != "system":
                prompt_messages.insert(0, {"role": "system", "content": ""})
            else:
                prompt_messages.insert(0, example["chosen"][0])
            # TODO: handle case where chosen/rejected also have system messages
            chosen_messages = example["chosen"][1:]
            rejected_messages = example["rejected"][1:]
            example["text_chosen"] = tokenizer.apply_chat_template(chosen_messages, tokenize=False)
            example["text_rejected"] = tokenizer.apply_chat_template(rejected_messages, tokenize=False)
            example["text_prompt"] = tokenizer.apply_chat_template(
                prompt_messages, tokenize=False, add_generation_prompt=True
            )
            example["text_chosen"] = _strip_prefix(example["text_chosen"], assistant_prefix)
            example["text_rejected"] = _strip_prefix(example["text_rejected"], assistant_prefix)
    else:
        raise ValueError(
            f"Could not format example as dialogue for `dpo` task! Require `[chosen, rejected]` keys but found {list(example.keys())}"
        )
    return example
column_names = list(raw_datasets["train"].features)
raw_datasets = raw_datasets.map(
        apply_chat_template,
        fn_kwargs={"tokenizer": tokenizer},
        num_proc=cpu_count(),
        remove_columns=column_names,
        desc="Formatting comparisons with prompt template",
)
for split in ["train", "test"]:
    raw_datasets[split] = raw_datasets[split].rename_columns(
        {"text_prompt": "prompt", "text_chosen": "chosen", "text_rejected": "rejected"}
    )
# Print a few random samples from the training set:
for index in random.sample(range(len(raw_datasets["train"])), 3):
    print(f"Prompt sample {index} of the raw training set:\n\n{raw_datasets['train'][index]['prompt']}")
    print(f"Chosen sample {index} of the raw training set:\n\n{raw_datasets['train'][index]['chosen']}")
    print(f"Rejected sample {index} of the raw training set:\n\n{raw_datasets['train'][index]['rejected']}")
peft_config = PeftConfig.from_pretrained(
    "./mistral-7b"
)
print("Adapter weights model repo:", model_id)
print("Base model weights model repo:", peft_config.base_model_name_or_path)
# Setting the local model path
base_model_path = "./mistral-7b-base"  # Base Model Path
adapter_path = "./mistral-7b"          # Adapter Weighting Path
# Load Peft configuration
peft_config = PeftConfig.from_pretrained(adapter_path)
# Print model information
print("Adapter weights model path:", adapter_path)
print("Base model weights model path:", peft_config.base_model_name_or_path)
# Specify how to quantify the model
quantization_config = BitsAndBytesConfig(
    load_in_4bit=False,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
# Load base model (Mistral-7B)
model_kwargs = {
    "torch_dtype": torch.bfloat16,
    "use_cache": False,
    "trust_remote_code": True,
    "local_files_only": True,
    "use_safetensors": False,
}
# Add quantization configuration only when quantization is required
if quantization_config.load_in_4bit:
    model_kwargs["quantization_config"] = quantization_config
    print("Enable 4-bit quantized loading model")
else:
    print("Not using quantitative loading models")
# load model (no device_map specified, managed by DeepSpeed)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    **model_kwargs
)
model = PeftModel.from_pretrained(
    base_model,
    adapter_path,
    local_files_only=True
)
print("Model has been successfully loaded locally!")
for name, param in model.named_parameters():
  print(name, param.requires_grad)
output_dir = 'data/zephyr-7b-dpo-lora'
training_args = TrainingArguments(
    # Basic training configuration
    output_dir=output_dir,
    num_train_epochs=1,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=2,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    learning_rate=5.0e-6,
    optim="paged_adamw_32bit",
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    # Evaluate configurations
    do_eval=True,
    eval_strategy="steps",
    eval_steps=25,
    # Logging and saving configurations
    logging_steps=10,
    save_strategy="steps",
    save_steps=25,
    save_total_limit=1,
    log_level="info",
    # System configuration
    bf16=True,
    seed=42,
    hub_model_id="zephyr-7b-dpo-qlora",
    # Multi-GPU configurations
    dataloader_num_workers=4,
    ddp_find_unused_parameters=False
)
# Add additional parameters required by DPOTrainer
dpo_args = {
    "beta": 0.1,
    "generate_during_eval": False,
    "padding_value": tokenizer.pad_token_id or 0,
    "label_pad_token_id": tokenizer.pad_token_id or 0,
    "max_prompt_length": 512,
    "max_completion_length": 512,
    "max_length": 1024,
    "truncation_mode": "keep_end",
    "disable_dropout": True,
    "use_liger_loss": False,
    "precompute_ref_log_probs": False,
    "use_logits_to_keep": False,
    "padding_free": False,
    "loss_type": "sigmoid",
    "label_smoothing": 0.0,
    "use_weighting": False,
    "f_divergence_type": "kl",
    "f_alpha_divergence_coef": 1.0,
    "dataset_num_proc": 4,
    "tools": None,
    "sync_ref_model": False,
    "tr_dpo": False,
    "force_use_ref_model": False,
    "rpo_alpha": None,
    "ld_alpha": 0.0,
    "ref_model_sync_steps": 0,
    "ref_model_mixup_alpha": 0.0,
    "model_init_kwargs": None,
    "ref_model_init_kwargs": None,
    "model_adapter_name": None,
    "ref_adapter_name": None,
    "reference_free": False
}
# Add DPO parameters to training_args
for key, value in dpo_args.items():
    setattr(training_args, key, value)
# LoRA configuration
peft_config = LoraConfig(
    r=128,
    lora_alpha=128,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
)
# Initialize the DPO trainer
trainer = DPOTrainer(
    model=model,
    ref_model=None,
    args=training_args,
    train_dataset=raw_datasets["train"],
    eval_dataset=raw_datasets["test"],
    processing_class=tokenizer, 
    peft_config=peft_config,
)
# Initiate training
train_result = trainer.train()
#Save Model
metrics = train_result.metrics
metrics["train_samples"] = len(raw_datasets["train"])
trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)
trainer.save_state()
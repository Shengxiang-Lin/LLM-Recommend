from datasets import load_dataset, DatasetDict
import re
import random
from multiprocessing import cpu_count
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
import torch
import os
from trl import SFTTrainer
from peft import LoraConfig
# based on config
raw_datasets = load_dataset("./ultrachat_200k")
print("Basic information about the dataset:")
print(raw_datasets["train_sft"])
print(raw_datasets["test_sft"])
indices = range(0,1000)
dataset_dict = {"train": raw_datasets["train_sft"].select(indices),
                "test": raw_datasets["test_sft"].select(indices)}
raw_datasets = DatasetDict(dataset_dict)
example = raw_datasets["train"][0]
print(example.keys())
print("\nThe content of each field:")
print(f"prompt: {example['prompt']}")
print(f"prompt_id: {example['prompt_id']}")
print(f"messages: {example['messages']}")
messages = example["messages"]
for message in messages:
  role = message["role"]
  content = message["content"]
  print('{0:20}:  {1}'.format(role, content))
model_id = "mistralai/Mistral-7B-v0.1"
# Load from local path
tokenizer = AutoTokenizer.from_pretrained("./mistral-7b")
# set pad_token_id equal to the eos_token_id if not set
if tokenizer.pad_token_id is None:
  tokenizer.pad_token_id = tokenizer.eos_token_id
# Set reasonable default for models without max length
if tokenizer.model_max_length > 100_000:
  tokenizer.model_max_length = 2048
# Set chat template
DEFAULT_CHAT_TEMPLATE = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"
tokenizer.chat_template = DEFAULT_CHAT_TEMPLATE
def apply_chat_template(example, tokenizer):
    messages = example["messages"]
    # We add an empty system message if there is none
    if messages[0]["role"] != "system":
        messages.insert(0, {"role": "system", "content": ""})
    example["text"] = tokenizer.apply_chat_template(messages, tokenize=False)
    return example

column_names = list(raw_datasets["train"].features)
raw_datasets = raw_datasets.map(apply_chat_template,
                                num_proc=cpu_count(),
                                fn_kwargs={"tokenizer": tokenizer},
                                remove_columns=column_names,
                                desc="Applying chat template",)
# create the splits
train_dataset = raw_datasets["train"]
eval_dataset = raw_datasets["test"]
for index in random.sample(range(len(raw_datasets["train"])), 3):
  print(f"Sample {index} of the processed training set:\n\n{raw_datasets['train'][index]['text']}")
# specify how to quantize the model
quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype="bfloat16",
)
#device_map = {"": torch.cuda.current_device()} if torch.cuda.is_available() else None
device_map = "auto"
#device_map = {"": 0} 
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# Direct loading of models
model_kwargs = dict(
    attn_implementation="flash_attention_2", # set this to True if your GPU supports it (Flash Attention drastically speeds up model computations)
    torch_dtype="auto",
    use_cache=False, # set to False as we're going to use gradient checkpointing
    device_map=device_map,
    quantization_config=quantization_config,
)
local_model_path = "./mistral-7b"  # The model and classifier files are in this directory
tokenizer = AutoTokenizer.from_pretrained(local_model_path)  # Locally Loaded Splitters
model = AutoModelForCausalLM.from_pretrained(
    local_model_path,  # local loading model
    **model_kwargs
)
# path where the Trainer will save its checkpoints and logs
output_dir = 'data/zephyr-7b-sft-lora'
# based on config
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
    per_device_eval_batch_size=4, # originally set to 8
    per_device_train_batch_size=4, # originally set to 8
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

train_result = trainer.train()
# Save Model
metrics = train_result.metrics
metrics["train_samples"] = len(train_dataset)
trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)
trainer.save_state()
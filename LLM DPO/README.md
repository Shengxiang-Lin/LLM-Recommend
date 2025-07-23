# Dataset and Model Download   
  
Dataset download (ultrafeedback_binarized)   
```bash  
sudo apt install git-lfs    
git clone https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized
```     
Model download (Zephyr-7B-SFT-LoRA model)  
Create a Python download script (e.g., download_zephyr_lora.py):  
```bash  
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="alignment-handbook/zephyr-7b-sft-lora",
    local_dir="./mistral-7b",
    repo_type="model",
    revision="main",
    token="XXX",  
    force_download=True,
    max_workers=4, 
    resume_download=True
)
```
Run script download
```bash
python download_zephyr_lora.py 
```
The results are as follows   
```bash  
README.md: 1.64kB [00:00, 3.96MB/s]                                                  | 0/21 [00:00<?, ?it/s]
.gitattributes: 1.52kB [00:00, 3.99MB/s]
adapter_config.json: 100%|█████████████████████████████████████████████████| 655/655 [00:00<00:00, 2.62MB/s]
all_results.json: 100%|████████████████████████████████████████████████████| 365/365 [00:00<00:00, 1.69MB/s]
config.json: 1.03kB [00:00, 3.52MB/s]                                             | 0.00/655 [00:00<?, ?B/s]
eval_results.json: 100%|████████████████████████████████████████████████████| 188/188 [00:00<00:00, 695kB/s]
(…)nts.1704718276.ip-26-0-162-180.3383747.0: 100%|█████████████████████| 72.6k/72.6k [00:00<00:00, 1.29MB/s]
(…)nts.1704730231.ip-26-0-162-180.3383747.1: 100%|█████████████████████████| 359/359 [00:00<00:00, 1.44MB/s]
(…)ents.1704763362.ip-26-0-163-236.395917.1: 100%|█████████████████████████| 359/359 [00:00<00:00, 1.43MB/s]
(…)nts.1699532512.ip-26-0-147-204.1177357.0: 100%|█████████████████████| 4.43k/4.43k [00:00<00:00, 13.6MB/s]
(…)ents.1704751173.ip-26-0-163-236.395917.0: 100%|██████████████████████| 73.8k/73.8k [00:00<00:00, 179kB/s]
(…)nts.1699533674.ip-26-0-147-204.1185167.0: 100%|█████████████████████| 12.0k/12.0k [00:00<00:00, 33.8MB/s]
(…)ents.1699572446.ip-26-0-155-174.110015.1: 100%|█████████████████████████| 359/359 [00:00<00:00, 1.61MB/s]
(…)ents.1699561511.ip-26-0-155-174.110015.0: 100%|█████████████████████| 13.5k/13.5k [00:00<00:00, 64.2MB/s]
special_tokens_map.json: 100%|█████████████████████████████████████████████| 437/437 [00:00<00:00, 1.70MB/s]
tokenizer.json: 1.80MB [00:00, 3.48MB/s]5.0:   0%|                              | 0.00/13.5k [00:00<?, ?B/s]
tokenizer_config.json: 1.39kB [00:00, 3.51MB/s]                                   | 0.00/437 [00:00<?, ?B/s]
train_results.json: 100%|███████████████████████████████████████████████████| 197/197 [00:00<00:00, 925kB/s]
trainer_state.json: 53.6kB [00:00, 876kB/s]
training_args.bin: 100%|███████████████████████████████████████████████| 4.79k/4.79k [00:00<00:00, 15.9MB/s]
adapter_model.safetensors: 100%|████████████████████████████████████████| 83.9M/83.9M [05:19<00:00, 263kB/s]
Fetching 21 files: 100%|████████████████████████████████████████████████████| 21/21 [05:20<00:00, 15.28s/it]
```    
Foundation model (Mistral-7B-v0.1) download
```bash  
# Set mirror source (temporary effect, valid only for the current terminal)
export HF_ENDPOINT=https://hf-mirror.com
# Execute download command
huggingface-cli download mistralai/Mistral-7B-v0.1 --local-dir ./mistral-7b-base --local-dir-use-symlinks False
```
The results are as follows   
```bash  
config.json: 571B [00:00, 1.59MB/s]                                                                         
.gitattributes: 1.52kB [00:00, 4.25MB/s]                                                                    
model.safetensors.index.json: 25.1kB [00:00, 39.6MB/s]                                                      
README.md: 1.55kB [00:00, 4.44MB/s]                                                                         
generation_config.json: 100%|███████████████████████████████████████████████| 116/116 [00:00<00:00, 439kB/s]
pytorch_model.bin.index.json: 23.9kB [00:00, 37.5MB/s]                                                      
special_tokens_map.json: 100%|█████████████████████████████████████████████| 414/414 [00:00<00:00, 1.58MB/s]
tokenizer.json: 1.80MB [00:00, 2.47MB/s]                                        | 0.00/1.09k [00:00<?, ?B/s]
tokenizer_config.json: 996B [00:00, 2.66MB/s]                                                               
tokenizer.model: 100%|████████████████████████████████████████████████████| 493k/493k [00:00<00:00, 518kB/s]
pytorch_model-00002-of-00002.bin: 100%|███████████████████████████████| 5.06G/5.06G [2:06:22<00:00, 668kB/s]
pytorch_model-00001-of-00002.bin: 100%|██████████████████████████████| 9.94G/9.94G [2:14:20<00:00, 1.23MB/s]
Fetching 12 files: 100%|█████████████████████████████████████████████████| 12/12 [2:14:23<00:00, 671.97s/it]
``` 
File directory structure     
```bash
Current working directory/
├─ mistral-7b/                  # Zephyr-7B-SFT-LoRA model (including LoRA adapter)
├─ mistral-7b-base/             # Mistral-7B-v0.1 Base Model
└─ ultrafeedback_binarized/     # ultrafeedback_binarized preference dataset
```

下载 ultrachat_200k
```bash  
sudo apt install git-lfs    
git clone https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k
```   
下载 mistralai/Mistral-7B-v0.1,新建.py文件,运行如下代码   
```bash  
from huggingface_hub import snapshot_download    
snapshot_download(        
    repo_id="mistralai/Mistral-7B-v0.1",       
    local_dir="./mistral-7b",        
    repo_type="model",          
    revision="main",          
    token="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",          
    force_download=True,         
    max_workers=4,        
    resume_download=True        
)
```  
运行结果如下
```bash  
.gitattributes: 100%|██████████████████████████████████████████████████| 1.52k/1.52k [00:00<00:00, 7.37MB/s]
README.md: 100%|███████████████████████████████████████████████████████| 1.55k/1.55k [00:00<00:00, 8.33MB/s]
generation_config.json: 100%|███████████████████████████████████████████████| 116/116 [00:00<00:00, 665kB/s]
config.json: 100%|█████████████████████████████████████████████████████████| 571/571 [00:00<00:00, 3.21MB/s]
model.safetensors.index.json: 100%|████████████████████████████████████| 25.1k/25.1k [00:00<00:00, 83.4kB/s]
model-00002-of-00002.safetensors: 100%|████████████████████████████████| 4.54G/4.54G [25:18<00:00, 2.99MB/s]
pytorch_model.bin.index.json: 100%|█████████████████████████████████████| 23.9k/23.9k [00:00<00:00, 588kB/s]
special_tokens_map.json: 100%|█████████████████████████████████████████████| 414/414 [00:00<00:00, 1.77MB/s]
tokenizer.json: 100%|██████████████████████████████████████████████████| 1.80M/1.80M [00:00<00:00, 1.96MB/s]
tokenizer.model: 100%|███████████████████████████████████████████████████| 493k/493k [00:00<00:00, 1.92MB/s]
tokenizer_config.json: 100%|███████████████████████████████████████████████| 996/996 [00:00<00:00, 5.62MB/s]
pytorch_model-00002-of-00002.bin: 100%|████████████████████████████████| 5.06G/5.06G [43:12<00:00, 1.95MB/s]
pytorch_model-00001-of-00002.bin: 100%|████████████████████████████████| 9.94G/9.94G [46:14<00:00, 3.58MB/s]
model-00001-of-00002.safetensors: 100%|████████████████████████████████| 9.94G/9.94G [52:53<00:00, 3.13MB/s]
Fetching 14 files: 100%|███████████████████████████████████████████████████| 14/14 [52:55<00:00, 226.80s/it]
```
下载完成   
此时mistral-7b、ultrachat_200、Supervised_fine_tuning_(SFT)_of_an_LLM_using_Hugging_Face_tooling.ipynb在同一层目录下  

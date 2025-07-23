# Configure Deepspeed    

Install the accelerate and deepspeed libraries   
```bash
pip install accelerate   
pip3 install deepspeed  
```
Generate initial configuration file
```bash
python -c "from accelerate.utils import write_basic_config; write_basic_config(mixed_precision='fp16')"
```  
View configuration     
```bash
accelerate env 
```
Test
```bash
accelerate test
```    
Relevant content in .cache/huggingface/accelerate/default_config.yaml  
```bash
{
  "compute_environment": "LOCAL_MACHINE",
  "debug": false,
  "deepspeed_config":{
    "gradient_clipping": 1.0,
    "zero_stage": 2
  },
  "distributed_type": "DEEPSPEED",
  "downcast_bf16": false,
  "enable_cpu_affinity": false,
  "machine_rank": 0,
  "main_training_function": "main",
  "mixed_precision": "no",
  "num_machines": 1,
  "num_processes": 2,
  "rdzv_backend": "static",
  "same_network": false,
  "tpu_use_cluster": false,
  "tpu_use_sudo": false,
  "use_cpu": false
}
```
Run demo.py   
```bash
python -m accelerate.commands.launch demo.py 
```


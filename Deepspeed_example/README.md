配置 deepspeed

pip3 install deepspeed
生成文件
python -c "from accelerate.utils import write_basic_config; write_basic_config(mixed_precision='fp16')"
查看配置
accelerate env
测试
accelerate test

相关配置
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


# Llama3 Experiments

## Install Environment
Install Package
```
conda create -n llama3_accessibility python=3.10 -y
conda activate llama3_accessibility
pip install --upgrade pip  # enable PEP 660 support
cd accessible-urban/llama3_experiments
pip install -e .
```

Install additional packages for training
```
# Make sure Your GPU driver is installed, check with `nvidia-smi` and `echo $CUDA_HOME`
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
```



## Preparation
1. Transfer the original training data into LLaVA format with 
```
cd accessible-urban/llama3_experiments
mkdir data & mkdir checkpoints
python preprocess_training_data.py
```

2. Download llama3-8b-instruct from [HuggingFace](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)

3. Download trained checkpoint `accessibility_llama3-8b-lora32_alpha16_ep10_bs64_lr3e-5` [here](https://usfedu-my.sharepoint.com/personal/lingyaol_usf_edu/_layouts/15/onedrive.aspx?ga=1&id=%2Fpersonal%2Flingyaol%5Fusf%5Fedu%2FDocuments%2FProject%20%2D%20Accessible), and unzip it under `checkpoints` directory.


## Training
Llama3-8b LoRA finetuning with flash attention requires 40GB GPUs with CUDA >11.7 (see more [details](https://github.com/Dao-AILab/flash-attention?tab=readme-ov-file#nvidia-cuda-support))

```
./scripts/finetune_lora.sh
```

## Evaluation
You can use our trained checkpoint  `accessibility_llama3-8b-lora32_alpha16_ep10_bs64_lr3e-5` or new checkpoint trained by your own.
```
./scripts/evaluate.sh'
```

You will get 
```
              precision    recall  f1-score   support

    negative       0.90      0.86      0.88       127
     neutral       0.75      0.52      0.62        23
    positive       0.95      0.89      0.92       129
   unrelated       0.90      0.97      0.93       289

    accuracy                           0.91       568
   macro avg       0.88      0.81      0.84       568
weighted avg       0.91      0.91      0.90       568
```



## Acknowledgement
This code is adapted from awesome projects: [LLaVA](https://github.com/haotian-liu/LLaVA) and [Open-LLaVA-NeXT](https://github.com/xiaoachen98/Open-LLaVA-NeXT)
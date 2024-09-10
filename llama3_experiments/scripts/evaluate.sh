# Evaluate on one GPU (at least 18GB GPU memory)
MODEL_NAME=accessibility_llama3-8b-lora32_alpha16_ep10_bs64_lr3e-5 # our best model checkpoint
MODEL_BASE=<Path_to_your_downloaded_Meta-Llama-3-8B-Instruct-HF> # downloaded from Hugging Face
CUDA_VISIBLE_DEVICES=1 python eval.py --model-path checkpoints/${MODEL_NAME} --model-base ${MODEL_BASE} --test-files ../test.jsonl  --key target_text

python metric.py --groundtruth-file ../test.jsonl --prediction-file data/test-prediction.jsonl
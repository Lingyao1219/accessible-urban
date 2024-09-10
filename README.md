# accessible-urban

## LLama3 model 
For Llama 3, we select the Llama-3-8B-Instruct as the backbone model and fine-tune it with LoRA.
The setting for the best checkpoint is:
|Parameter|number|
|--|--|
|Epoch|10|
|Lora Rank|32|
|lr|3e-5|
|batch size|64|

We obtain 91% accuracy on the test dataset.

More details can be found in `llama3_experiments` directory.
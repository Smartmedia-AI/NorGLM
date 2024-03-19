import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, EvalPrediction
from peft import LoraConfig, get_peft_model 
import transformers
from datasets import load_dataset
from scipy.special import softmax
import numpy as np


tokenizer_id_or_path = "NorGLM-369M/NorGLM-369M"
tokenizer_max_len = 2048
tokenizer_config = {'pretrained_model_name_or_path': tokenizer_id_or_path,
                            'max_len': tokenizer_max_len}
tokenizer = AutoTokenizer.from_pretrained(**tokenizer_config)
tokenizer.pad_token = tokenizer.eos_token

model_config = AutoConfig.from_pretrained("NorGLM-369M/NorGLM-369M")
model = AutoModelForCausalLM.from_pretrained(
    "NorGLM-369M/NorGLM-369M", 
    device_map='auto',
    config=model_config)

for param in model.parameters():
  param.requires_grad = False  # freeze the model - train adapters later
  if param.ndim == 1:
    # cast the small parameters (e.g. layernorm) to fp32 for stability
    param.data = param.data.to(torch.float32)

model.gradient_checkpointing_enable()  # reduce number of stored activations
model.enable_input_require_grads()

class CastOutputToFloat(nn.Sequential):
  def forward(self, x): return super().forward(x).to(torch.float32)
model.lm_head = CastOutputToFloat(model.lm_head)

def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}")
    

config = LoraConfig(
    r=16, #attention heads, rank of the attention matrix, i think
    lora_alpha=32, #alpha scaling
    # target_modules=["q_proj", "v_proj"], #will be set after i know the names
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM" # set this for CLM or Seq2Seq
)


model = get_peft_model(model, config)
print_trainable_parameters(model)

# dataNbAiLab = load_dataset("NbAiLab/norwegian-alpaca", split='train[:80%]')

print("LOADING 80% OF DATASET")
dataALl = load_dataset("./conversation")
data = dataALl["train"]
print("LOADING LAST 20% OF DATASET FOR EVALUATING")
eval_data = dataALl["test"]

def merge_columns(example):
    example["prediction"] = str(example["prompt"]) + " : " + str(example["answer"])
    return example
     
data = data.map(merge_columns)
data = data.map(lambda samples: tokenizer(samples['prediction'], padding=True, truncation=True, max_length=tokenizer_max_len), batched=True)

eval_data = eval_data.map(merge_columns)
eval_data = eval_data.map(lambda samples: tokenizer(samples['prediction'], padding=True, truncation=True, max_length=tokenizer_max_len), batched=True)

data = data.remove_columns(["id", "turn_id"])
eval_data = eval_data.remove_columns(["id", "turn_id"])

print("SETTING TRAINER")
trainer = transformers.Trainer(
    model=model, 
    train_dataset=data,
    args=transformers.TrainingArguments(
        per_device_train_batch_size=4, 
        gradient_accumulation_steps=1,
        num_train_epochs=10,
        warmup_steps=100, 
        learning_rate=9e-6,
        #fp16=True,
        logging_steps=1, 
        output_dir='./results/Checkpoints_Peft_NorGLM369M',
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    eval_dataset=eval_data,
)

model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
print("STARTING TRAINING:...")
trainer.train()


peft_model_id="./results/PEFT_NorGLM369M"
trainer.model.save_pretrained(peft_model_id)
tokenizer.save_pretrained(peft_model_id)

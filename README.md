# NLEBench + NorGLM

This github repo is for the resources related to our paper: NLEBench+NorGLM: A Comprehensive Empirical Analysis and Benchmark Dataset for Generative Language Models in Norwegian

Data in the /results are used for reproducing the results in our paper.

# More Information

For open source and easy access, we have put all relevant resources on Hugging Face. Click on [Hugginface Link](https://huggingface.co/NorGLM) to find detailed information on the following:
1) The NLEBench datasets and Dataset cards
2) NorGLMs and Model cards
3) Lora fine-tuned adaptors of NorGLMs for different downstream tasks
4) Evaluation tool - pretrained entailment model and model inference examples

## Avalable resources
NorGLM models are trained from scratch with nearly 25B tokens including Norwegian, Danish, Swedish, German and English languages. NorGLM incorporates the following models: 

| Models          | #parameters | base model |
| :---------------- | :------: | :----: |
| NorGPT-369M      |   369M   | GPT2 |
| NorGPT-3B       |   3B   | GPT2 |
| NorLlama-3B    |  3B   | Llama |
| NorGPT-23B |  23B   | GPT2 |

We also trained a series of Norwegian foundation models based on state-of-the-art open models at [NorLLM Huggingface Link](https://huggingface.co/NorwAI).

| Models          | #parameters | base model |
| :---------------- | :------: | :----: |
| NorwAI-Mistral-7B      |   7B   | Mistral-v1 |
| NorwAI-Mistral-7B-pretrain   |   7B   | Mistral-v1 |
| NorwAI-Llama2-7B    |  7B   | Llama2 |
| NorwAI-Mixtral-8x7B |  45B   | Mixtral-8x7B |
| NorwAI-Mistral-7B-instruct  |   7B   | Mistral-v1 |
| NorwAI-Mixtral-8x7B-instruct |  45B   | Mixtral-8x7B |


## LICENSE

All of our NorGLM models and NLEBench datasets following the CC BY-NC-SA 4.0 DEED license and can only be used for research purpose.

## Cite Us

If you feel our work is helpful, please cite our paper:

```
@article{liu2023nlebench+,
  title={NLEBench+ NorGLM: A Comprehensive Empirical Analysis and Benchmark Dataset for Generative Language Models in Norwegian},
  author={Liu, Peng and Zhang, Lemei and Farup, Terje Nissen and Lauvrak, Even W and Ingvaldsen, Jon Espen and Eide, Simen and Gulla, Jon Atle and Yang, Zhirong},
  journal={arXiv preprint arXiv:2312.01314},
  year={2023}
}
```


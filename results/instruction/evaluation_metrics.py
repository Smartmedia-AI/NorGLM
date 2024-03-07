import pandas as pd
from transformers import AutoTokenizer, AutoModelForMaskedLM, BertForSequenceClassification
from transformers import LlamaTokenizer, LlamaForCausalLM, AutoConfig
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sacrebleu.metrics import BLEU
from rouge import Rouge
from math import log
import nltk
from nltk.util import ngrams
import mauve
import matplotlib.pyplot as plt
import numpy as np
from evaluate import load
import cupy
from cupy.cuda import cublas
from datasets import load_dataset
import json
import os

handle = cublas.create()
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


#nltk.download('punkt')

def n_gram(text, num):
	#tokens = nltk.word_tokenize(text)
	ngs = ngrams(nltk.word_tokenize(text), num)
	ngram_list = list(ngs)
	return ngram_list

def entropy(texts, num, base=None):
	# Entropy of n-gram input text
	ngram_list = n_gram(texts, num)
	fdist = nltk.FreqDist(ngram_list)
	n_tokens = len(fdist)
	if n_tokens <= 1:
		return 0

	freqs = [item[1] for item in fdist.items()]
	probs = np.array(freqs) / float(n_tokens)
	base = 'e' if base is None else base
	ent = 0.
	ngram_probs = []
	for i in probs:
		if base == 'e':
			ent -= i * log(i)
		else:
			ent -= i * log(i,base)
	return ent

def entr_score(hypothesis, base=None):
	# hypothesisi is a list of generated outputs
	# ENTR is to calculate the geometric mean of the entropy of unigram,
	# bigram and trigram of the machine generated outputs
	unigram_ent = entropy(hypothesis, 1)
	bigram_ent = entropy(hypothesis, 2)
	trigram_ent = entropy(hypothesis, 3)
	if base==None:
		gmean = np.exp(np.log([unigram_ent, bigram_ent, trigram_ent]).mean())
	else:
		gmean = np.exp(np.log([unigram_ent, bigram_ent, trigram_ent], base).mean())

	return gmean

def informativeness_score_document_level(text):
	# Distinct-n metric
    # outputs is a list which contains several sentences, each sentence contains several wordsex
    tokens = nltk.word_tokenize(text)
    unigram_list = n_gram(text, 1)
    unigram_fdist = nltk.FreqDist(unigram_list)
    bigram_list = n_gram(text, 2)
    bigram_fdist = nltk.FreqDist(bigram_list)
    trigram_list = n_gram(text, 3)
    trigram_fdist = nltk.FreqDist(trigram_list)
    quagram_list = n_gram(text, 4)
    quagram_fdist = nltk.FreqDist(quagram_list)
    sums_1 = [item[1] for item in unigram_fdist.items()]
    sums_2 = [item[1] for item in bigram_fdist.items()]
    sums_3 = [item[1] for item in trigram_fdist.items()]
    sums_4 = [item[1] for item in quagram_fdist.items()]
    
    dis1 = len(unigram_list) / float(len(tokens))
    dis2 = len(bigram_list) / float(len(tokens))
    dis3 = len(trigram_list)/float(len(tokens))
    dis4 = len(quagram_list)/float(len(tokens))
    return dis1, dis2, dis3, dis4

def rouge_score(hypothesis, references):
	# ROUGE score includes recall (r), precision (p) and harmonic mean value (f)
	rouge_scorer = Rouge()
	score = rouge_scorer.get_scores(
	    hyps=hypothesis,
	    refs=references,
	)
	result = score
	return result

def two_seq_same(sa, sb):
    if len(sa) != len(sb):
        return False
    for (wa, wb) in zip(sa, sb):
        if wa != wb:
            return False
    return True

def unique_sentence_ratio(sequence_batch):
	# Unique Sentence Ratio (USR) metric
	unique_seq = []
	for seq in sequence_batch:
		count = 0
		for uni_seq in unique_seq:
			if two_seq_same(seq, uni_seq):
				count += 1
				break
		if count == 0:
			unique_seq.append(seq)

	return len(unique_seq) / len(sequence_batch), len(unique_seq)

def bleu_score(hypothesis, references):
	bleu_scorer = BLEU()
	score = bleu_scorer.sentence_score(
	    hypothesis=hypothesis,
	    references=[references]
	)
	result = score.score
	return result

def mauve_score(p_text, q_text):
	#p_text: human  q_text: machine in list format
	# call mauve.compute_mauve using raw text on GPU 0; each generation is truncated to 256 tokens
	out = mauve.compute_mauve(p_text=p_text, q_text=q_text, device_id=0, max_text_length=256, verbose=False)
	# plot divergence curve  
	plt.plot(out.divergence_curve[:, 1], out.divergence_curve[:, 0])
	result = out.mauve
	return result

def eva_pairs(references, hypo_list):
	rouge1_list = []
	rouge2_list = []
	rougel_list = []
	bleu_list = []
	usr_list = []
	dist_1_list = []
	dist_2_list = []
	dist_3_list = []
	dist_4_list = []
	entr_list = []
	mauve = 0.

	for ref, hypo in zip(references, hypo_list):
		bleu = bleu_score(hypo, ref)
		try:
			rouge = rouge_score(hypo, ref)
		except Exception as e:
			print(e)
			continue
		dist_1, dist_2, dist_3, dist_4 = informativeness_score_document_level(hypo)
		entr = entr_score(hypo)
		rouge1_list.append(rouge[0]['rouge-1']['f'])
		rouge2_list.append(rouge[0]['rouge-2']['f'])
		rougel_list.append(rouge[0]['rouge-l']['f'])
		bleu_list.append(bleu)
		dist_1_list.append(dist_1)
		dist_2_list.append(dist_2)
		dist_3_list.append(dist_3)
		dist_4_list.append(dist_4)
		entr_list.append(entr)

	bleu = np.array(bleu_list).mean()
	rouge1 = np.array(rouge1_list).mean()
	rouge2 = np.array(rouge2_list).mean()
	rougel = np.array(rougel_list).mean()
	dist_1 = np.array(dist_1_list).mean()
	dist_2 = np.array(dist_2_list).mean()
	dist_3 = np.array(dist_3_list).mean()
	dist_4 = np.array(dist_4_list).mean()
	entr = np.array(entr_list).mean()
	usr = unique_sentence_ratio(hypo_list)
	mauve = mauve_score(references, hypo_list)
	print("BLEU score is {}".format(bleu))
	print("ROUGE-1 score is {}".format(rouge1))
	print("ROUGE-2 score is {}".format(rouge2))
	print("ROUGE-L score is {}".format(rougel))
	print("Distinct-1 score is {}".format(dist_1))
	print("Distinct-2 score is {}".format(dist_2))
	print("Distinct-3 score is {}".format(dist_3))
	print("Distinct-4 score is {}".format(dist_4))
	print("USR score is {}".format(usr))
	print("ENTR score is {}".format(entr))
	print("MAUVE score is {}".format(mauve))
	print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
	return bleu,rouge1,rouge2,rougel,dist_1,dist_2,dist_3,dist_4,entr,usr,mauve

def entailment_score(texts, references, generated_texts, model_id):
	# Entailment: 1, Contradict: 0, Neutral: 2
	# concatinate news articles and generated summaries as input
	input_texts = [t + ' [SEP] '+ g for t,g in zip(texts, generated_texts)]
	input_texts = [text for text in input_texts if len(text.split())<1024]
	# Set the maximum sequence length.
	MAX_LEN = 512
	batch_size = 16

	dev = "cuda:0" if torch.cuda.is_available() else "cpu"
	dev = "cpu"
	device = torch.device(dev)
	#device = "mps" if torch.backends.mps.is_available() else "cpu"
	#device = torch.device(device)

	if 'Llama' in model_id:
		model_and_tokenizer_path = '/cluster/home/penl/workspace/TencentPretrain/llama/checkpoints/output_model.bin/NorLlama_3B_HF'
		tokenizer_max_len = 2048
		tokenizer_config = {'pretrained_model_name_or_path': model_and_tokenizer_path,
		                            'max_len': tokenizer_max_len}
		tokenizer = LlamaTokenizer.from_pretrained(tokenizer_id_or_path)
	
	elif '23B' in model_id:
		model_and_tokenizer_path = '/cluster/home/penl/workspace/checkpoints/NorGPT_20B/Megatron-HF/NorGPT_23B'
		tokenizer_max_len = 2048
		tokenizer_config = {'pretrained_model_name_or_path': model_and_tokenizer_path, 'max_len': tokenizer_max_len}
		tokenizer = AutoTokenizer.from_pretrained(**tokenizer_config, use_auth_token='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')
	
	    # model = AutoModelForCausalLM.from_pretrained(model_and_tokenizer_path, device_map='balanced', use_auth_token='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')
	    # #model.resize_token_embeddings(len(tokenizer))
	    # print("Load peft adaptor...")
	    # peft_model_id = "/cluster/home/penl/workspace/checkpoints/NorGPT_20B/Megatron-HF/ft_model_summarization/PEFT_NbAiLab6B"
	    # config = PeftConfig.from_pretrained(peft_model_id)
	    # model = PeftModel.from_pretrained(model, peft_model_id)


	# tokenizer = AutoTokenizer.from_pretrained("Entailment", fast_tokenizer=True)
	#tokenizer.pad_token = tokenizer.eos_token
	tokenizer.add_special_tokens({'pad_token': '[PAD]'})

	test_inputs = tokenizer(text=input_texts, add_special_tokens=True, return_attention_mask = True, return_tensors="pt", padding=True, truncation=True,  max_length=MAX_LEN)
	print('>>>>>>>>>>>>>>>>>>>>>>')
	print(test_inputs['input_ids'].size)
	validation_data = TensorDataset(test_inputs['input_ids'],test_inputs['attention_mask'])
	validation_dataloader = DataLoader(validation_data,batch_size=batch_size)

	model = BertForSequenceClassification.from_pretrained("Entailment")
	model.to(device)
	model.eval()

	results = []
	num_batches = 1
	for batch in validation_dataloader:
		# Add batch to GPU
		batch = tuple(t.to(device) for t in batch)
		# Unpack the inputs from our dataloader
		b_input_ids, b_input_mask = batch
		# Telling the model not to compute or store gradients, saving memory and speeding up validation
		with torch.no_grad():
			# Forward pass, calculate logit predictions
			logits = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)

		# Move logits and labels to CPU
		logits = logits[0].to('cpu').numpy()
		pred_flat = np.argmax(logits, axis=1).flatten()

		results.extend(pred_flat)
		num_batches += 1

	ent_ratio = results.count(1) / float(len(results))
	neu_ratio = results.count(2) / float(len(results))
	con_ratio = results.count(0) / float(len(results))
	print("Entailment ratio: {}; Neutral ratio: {}; Contradict ratio: {}.".format(ent_ratio, neu_ratio, con_ratio))
	return ent_ratio, neu_ratio, con_ratio

def perplexity_score(predictions, model_id):
	perplexity = load("perplexity", module_type="metric")
	if 'NbAiLab' in model_id:
		peft_model_id = '/cluster/home/penl/workspace/checkpoints/NorGPT_20B/Megatron-HF/fine_tuned_model/NbAiLab6B_eval/PEFT_NbAiLab6B'
		results = perplexity.compute(predictions=predictions, model_id=model_id, peft_model_id=peft_model_id, use_auth_token='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')
	elif 'continue' in model_id:
		peft_model_id = '/cluster/home/penl/workspace/checkpoints/NorGPT_20B/Megatron-HF/fine_tuned_model/NorGLM3B_continue_eval/PEFT_NorGLM3B_continue'
		model_id = '/cluster/home/penl/workspace/continue_pretrain/gpt2/HuggingFace/NorGPT-3B-continue'
		results = perplexity.compute(predictions=predictions, model_id=model_id, peft_model_id=peft_model_id)
	elif '23B' in model_id:
		peft_model_id = '/cluster/home/penl/workspace/checkpoints/NorGPT_20B/Megatron-HF/fine_tuned_model/PEFT_NorGLM23B'
		model_id = '/cluster/home/penl/workspace/checkpoints/NorGPT_20B/Megatron-HF/NorGPT_23B'
		results = perplexity.compute(predictions=predictions, model_id=model_id, peft_model_id=peft_model_id)
	elif 'Llama' in model_id:
		peft_model_id = '/cluster/home/penl/workspace/checkpoints/NorGPT_20B/Megatron-HF/fine_tuned_model/LLaMA3B_eval/PEFT_LLaMA3B'
		model_id = '/cluster/home/penl/workspace/TencentPretrain/llama/checkpoints/output_model.bin/NorLlama_3B_HF'
		results = perplexity.compute(predictions=predictions, model_id=model_id, peft_model_id=peft_model_id, is_llama=True)
	elif '369M' in model_id:
		peft_model_id = '/cluster/home/penl/workspace/checkpoints/NorGPT_20B/Megatron-HF/fine_tuned_model/NorGLM369M_eval/PEFT_NorGLM369M'
		model_id = '/cluster/home/penl/workspace/checkpoints/gpt2/Megatron-huggingface/NorGLM-369M'
		results = perplexity.compute(predictions=predictions, model_id=model_id, peft_model_id=peft_model_id)
	else: # 3B model
		peft_model_id = '/cluster/home/penl/workspace/checkpoints/NorGPT_20B/Megatron-HF/fine_tuned_model/NorGLM3B_eval/PEFT_NorGLM3B'
		model_id = '/cluster/home/penl/workspace/checkpoints/gpt2/Megatron-huggingface/NorGLM-3B'
		results = perplexity.compute(predictions=predictions, model_id=model_id, peft_model_id=peft_model_id)
	return results['mean_perplexity']

def main():
	#fpath = './data/'
	fnames = [#('text_23B.jsonl', 'NorGLM-23B'),
			#('text_Nb_6B.jsonl', 'NbAiLab/nb-gpt-j-6B'),
			#('text_369M.jsonl', 'NorGLM-369M'),
			#('text_Llama_3B.jsonl', 'NorLlama-3B'),
			#('text_3B.jsonl', 'NorGLM-3B'),
			#('text_3B_continue.jsonl', 'NorGLM-3B-continue')
			('instruction_GPT3.5.jsonl', 'GPT3.5')
			]
			
	fout = open("results_inst.txt", "a")

	#eval_data = load_dataset("NbAiLab/norwegian-alpaca", split='train[-20%:]')
	eval_data = load_dataset("csv",data_files="alpaca_test.csv")
	references = eval_data['train']['output']
	instruction = eval_data['train']['instruction']

	for fname in fnames:
		print("Reading file {}...".format(fname[0]))
		fout.write(fname[0] + "\n")

		data_list = []
		with open(fname[0], "r", encoding="utf-8-sig") as file:
		    for line in file:
		        data = json.loads(line)
		        data_list.append(data)

		df = pd.DataFrame(data_list)

		#df = pd.read_json(fname[0])
		remove_str = 'Token indices sequence length is longer than 2048.'
		df = df[df!=remove_str]
		df = df.dropna()
		#articles = df['article'].to_list()
		#references = df['positive_sample'].to_list()
		hypo_list1 = df['generated_text']
		# if ('NbAiLab' in fname[1]) or ('continue' in fname[1]) or ('23B' in fname[1]) or ('Llama' in fname[1]):
		# 	perplexity1 = perplexity_score(hypo_list1, fname[1])
		# else:
		# 	perplexity1 = perplexity_score(hypo_list1, fname[1])
		
		bleu1,rouge1,rouge2,rougel,dist_1,dist_2,dist_3,dist_4,entr,usr,mauve = eva_pairs(references, hypo_list1)
		#ent_ratio1, neu_ratio1, con_ratio1 = entailment_score(articles, references, hypo_list1, fname[1])
		#fout.write('Perplexity1 score is ' + str(perplexity1) + '\n')
		fout.write('BLEU score is ' + str(bleu1) + '\n')
		fout.write('ROUGE-1 score is ' + str(rouge1) + '\n')
		fout.write('ROUGE-2 score is ' + str(rouge2) + '\n')
		fout.write('ROUGE-L score is ' + str(rougel) + '\n')
		fout.write('Distinct-1 score is ' + str(dist_1) + '\n')
		fout.write('Distinct-2 score is ' + str(dist_2) + '\n')
		fout.write('Distinct-3 score is ' + str(dist_3) + '\n')
		fout.write('Distinct-4 score is ' + str(dist_4) + '\n')
		fout.write('USR score is ' + str(usr) + '\n')
		fout.write('ENTR score is ' + str(entr) + '\n')
		fout.write('MAUVE score is ' + str(mauve) + '\n')
		#fout.write('Entailment score is '+str(ent_ratio1)+' ; Neutral ratio is '+str(neu_ratio1)+' ; Condict ratio is '+str(con_ratio1)+'\n')

	fout.close()
	print("DONE!")


if __name__ == "__main__":
    main()
 







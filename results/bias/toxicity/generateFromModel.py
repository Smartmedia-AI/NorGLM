""" pip/pip3
pip install google-api-python-client
pip install translators
pip install transformers (or xformers)
"""

from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForCausalLM
import torch
import csv
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

### VARIABLES ###
MODEL_TEMPERATURE = 0.8  # default 1
MODEL_TOP_K = 0  # default 50
MODEL_TOP_P = 0.92  # default 1
MODEL_MIN_NEW_TOKENS = 50
MODEL_MAX_NEW_TOKENS = 100
MODEL_DO_SAMPLE = False
MODEL_NUM_BEAMS = 3
MODEL_NO_REPEAT_NGRAM_SIZE = 5

# def get_pipe(torch_device, model_id):
#     print('Creating pipeline for device: {}...'.format(torch_device))
#     model_and_tokenizer_path = model_id
#     tokenizer_max_len = 2048
#     tokenizer_config = {'pretrained_model_name_or_path': model_and_tokenizer_path, 'max_len': tokenizer_max_len}
#     tokenizer = AutoTokenizer.from_pretrained(**tokenizer_config)
#     model = AutoModelForCausalLM.from_pretrained(model_and_tokenizer_path, 
#         device_map='balanced',
#         pad_token_id=tokenizer.eos_token_id)
#     model.eval()
#     text_generator = pipeline("text-generation", 
#         model=model, 
#         tokenizer = tokenizer,
#         framework="pt",
#         device=torch_device)
#     return text_generator

def getCSVprompts():
    """return prompt, source, prompt_no"""
    # Path to your CSV file
    filepath = 'prompts_norwegian.csv'
    entry_ID = []
    prompt = []
    source = []
    prompt_no = []
    # Open the CSV file and populate the lists
    with open(filepath, 'r') as csvfile:
        reader = csv.reader(csvfile)
        # Skip the header
        #next(reader)
        for row in reader:
            entry_ID.append(row[0])
            prompt.append(row[1])
            source.append(row[2])
            prompt_no.append(row[3])
    # Printing out the lists
    return entry_ID, prompt, source, prompt_no

#text_generator = get_pipe(torch_device, model_id)
def generateNorwegianText(inputText):
    """Uses model to generate str text"""
    # sentence = text_generator("", min_new_tokens=10, max_new_tokens=20)
    sentence = text_generator(inputText, min_new_tokens=MODEL_MIN_NEW_TOKENS, 
        max_new_tokens=MODEL_MAX_NEW_TOKENS, 
        temperature=MODEL_TEMPERATURE,
        do_sample=MODEL_DO_SAMPLE, 
        num_beams=MODEL_NUM_BEAMS, 
        top_k=MODEL_TOP_K, 
        no_repeat_ngram_size=MODEL_NO_REPEAT_NGRAM_SIZE)
    text = sentence[0].get('generated_text')

    #STRIP IT:
    text_without_input = text.replace(inputText, "", 1).strip()
    return text_without_input

def generate_texts(model, tokenizer, prompt, min_new_tokens=20, max_seq_length=300, do_sample=True, top_p=0.95, top_k=10):
    model_inputs = tokenizer(prompt, return_tensors='pt').to(torch_device)
    output = model.generate(**model_inputs, 
        do_sample=False, 
        max_length = max_seq_length, 
        min_new_tokens=min_new_tokens, 
        no_repeat_ngram_size=2)
    result = tokenizer.decode(output[0], skip_special_tokens=True)
        #results.append(result)
    return result

torch_device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = '/cluster/home/penl/workspace/checkpoints/NorGPT_20B/Megatron-HF/NorGPT_23B'
model_and_tokenizer_path = model_id
tokenizer_max_len = 2048
tokenizer_config = {'pretrained_model_name_or_path': model_and_tokenizer_path, 'max_len': tokenizer_max_len}
tokenizer = AutoTokenizer.from_pretrained(**tokenizer_config)
model = AutoModelForCausalLM.from_pretrained(model_and_tokenizer_path, 
    device_map='balanced',
    pad_token_id=tokenizer.eos_token_id)
model.eval()

entry_ID, prompt, source, prompt_no = getCSVprompts()
generated_texts = ["generated_no_23B"]
with torch.no_grad():
    for i in range(len(entry_ID)):
        print("i: ", str(i))
        #for each entry, use the model to generate text
        if(i == 0):
            print("passing entry number 1 which is")
            print(entry_ID[i])
            print(prompt[i])
            print(source[i])
            print(prompt_no[i])
            pass

        generated_text = generate_texts(model, tokenizer, prompt_no[i])
        generated_texts.append(generated_text)
        
        with open('output_23B.csv', 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            # Write a row with the elements from the arrays
            csvwriter.writerow([entry_ID[i], prompt[i], source[i], prompt_no[i], generated_text])

print("DONE!")

"""
# Open a CSV file in write mode
with open('output.csv', 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    # Zip the arrays together to iterate over pairs of elements
    for item1, item2, item3, item4, item5 in zip(entry_ID, prompt, source, prompt_no, generated_texts):
        # Write a row with the two elements separated by a comma
        csvwriter.writerow([item1, item2, item3, item4, item5])
"""





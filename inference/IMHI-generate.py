import os
import argparse
import pandas as pd

import torch
from transformers import AutoModel, AutoTokenizer, LlamaForCausalLM, LlamaTokenizer, AutoModelForCausalLM
from peft import PeftModel

from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, classification_report, \
    precision_recall_fscore_support, precision_score, recall_score

def load_instruction_test_data():
    test_data = {}
    for root, ds, fs in os.walk("./test_instruction"):
        for fn in fs:
            data = pd.read_csv(os.path.join(root, fn))
            texts = data['query'].to_list()
            labels = data['gpt-3.5-turbo'].to_list()
            test_data[fn.split('.')[0]] = [texts, labels]
    return test_data

def load_complete_test_data():
    test_data = {}
    for root, ds, fs in os.walk("../test_data/test_complete"):
        for fn in fs:
            data = pd.read_csv(os.path.join(root, fn))
            texts = data['query'].to_list()
            labels = data['gpt-3.5-turbo'].to_list()
            test_data[fn.split('.')[0]] = [texts, labels]
    return test_data

def load_expert_data():
    test_data = {}
    for root, ds, fs in os.walk("../human_evaluation/test_instruction_expert"):
        for fn in fs:
            data = pd.read_csv(os.path.join(root, fn))
            texts = data['query'].to_list()
            labels = data['gpt-3.5-turbo'].to_list()
            test_data[fn.split('.')[0]] = [texts, labels]
    return test_data

def generate_response(model, tokenizer, test_data, device, batch_size):
    generated_text = {}
    goldens = {}

    #model.to(device)

    for dataset_name in test_data.keys():
        #if dataset_name not in ['DR', 'dreaddit']:
        #    continue
        print('Generating for dataset: {}'.format(dataset_name))
        queries, golden = test_data[dataset_name]
        goldens[dataset_name]  = golden
        responses = []

        for i in range(0, len(queries), batch_size):
            batch_data = queries[i: min(i+batch_size, len(queries))]
            #print(batch_data[:2])
            inputs = tokenizer(batch_data, return_tensors="pt", padding=True)
            #print(inputs)
            #final_input = inputs.input_ids
            input_ids = inputs.input_ids.to(device)
            attention_mask = inputs.attention_mask.to(device)
            #print(final_input)
            generate_ids = model.generate(input_ids, attention_mask=attention_mask, max_length=2048)
            for j in range(generate_ids.shape[0]):
                truc_ids = generate_ids[j][len(input_ids[j]) :]
                response = tokenizer.decode(truc_ids, skip_special_tokens=True, spaces_between_special_tokens=False)
                responses.append(response)
            print(i)
        generated_text[dataset_name] = responses

    return generated_text, goldens

def save_output(generated_text, goldens, output_path):
    if not os.path.exists("../IMHI_model_output/"):
        os.mkdir("../IMHI_model_output/")
    if not os.path.exists("../IMHI_model_output/"+output_path):
        os.mkdir("../IMHI_model_output/"+output_path)
    for dataset_name in generated_text.keys():
        output = {'goldens': goldens[dataset_name], 'generated_text': generated_text[dataset_name]}
        output = pd.DataFrame(output, index=None)
        output.to_csv("{}/{}/{}.csv".format('../IMHI_model_output',
                                         output_path, dataset_name), index=False, escapechar='\\')

def main(model_path: str, model_output_path: str, batch_size: int, test_dataset: str, rule_calculate: bool,
         llama: bool, device: str, lora: bool, cuda: bool, lora_path: str):
    if llama:
        model = LlamaForCausalLM.from_pretrained(model_path, device_map='auto', torch_dtype=torch.bfloat16, use_flash_attention_2=True)
        print(model.hf_device_map)
        tokenizer = LlamaTokenizer.from_pretrained(model_path, padding_side='left')
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, use_flash_attention_2=True)
        tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left')

    if lora:
        model = PeftModel.from_pretrained(model, lora_path)
        model = model.merge_and_unload()

    tokenizer.pad_token = tokenizer.unk_token

    if test_dataset == 'IMHI':
        test_data = load_instruction_test_data()
    elif test_dataset == 'IMHI-completion':
        test_data = load_complete_test_data()
    elif test_dataset == 'expert':
        test_data = load_expert_data()
    generated_text, goldens = generate_response(model, tokenizer, test_data, device, batch_size)
    save_output(generated_text, goldens, model_output_path)
    if rule_calculate:
        calculate_f1(generated_text, goldens, model_output_path)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='The IMHI benchmark.')
    parser.add_argument('--model_path', type=str, default='')
    parser.add_argument('--model_output_path', type=str)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--batch_size', type=int, default=24)
    parser.add_argument('--test_dataset', type=str, choices=['IMHI', 'IMHI-completion', 'expert'])
    parser.add_argument('--rule_calculate', action='store_true')
    parser.add_argument('--llama', action='store_true')
    parser.add_argument('--lora', action='store_true')
    parser.add_argument('--lora_path', type=str)

    args = parser.parse_args()
    args = vars(args)
    device = torch.device("cuda:0" if torch.cuda.is_available() and args['cuda'] is True else "cpu")
    args['device'] = device

    main(**args)

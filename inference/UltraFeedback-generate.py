import os
import argparse
import pandas as pd

import torch
from transformers import AutoModel, AutoTokenizer, LlamaForCausalLM, LlamaTokenizer, AutoModelForCausalLM
from peft import PeftModel

from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, classification_report, \
    precision_recall_fscore_support, precision_score, recall_score


def load_HH_RLHF_test_data():
    data = pd.read_csv('../dynamic_multi_objective_dataset/UltraFeedback-aligner-data/no_equal/test.csv')
    texts = data['query'].to_list()
    labels = data['golden_response'].to_list()
    test_data = [texts, labels]
    return test_data


def generate_response(model, tokenizer, test_data, device, batch_size):
    #model.to(device)
    queries, golden = test_data
    responses = []

    for i in range(0, len(queries), batch_size):
        batch_data = queries[i: min(i + batch_size, len(queries))]
        # print(batch_data[:2])
        inputs = tokenizer(batch_data, return_tensors="pt", padding=True)
        # print(inputs)
        # final_input = inputs.input_ids
        input_ids = inputs.input_ids.to(device)
        attention_mask = inputs.attention_mask.to(device)
        # print(final_input)
        generate_ids = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=728)
        for j in range(generate_ids.shape[0]):
            truc_ids = generate_ids[j][len(input_ids[j]):]
            response = tokenizer.decode(truc_ids, skip_special_tokens=True, spaces_between_special_tokens=False)
            responses.append(response)
        print(i)

    return queries, responses, golden


def save_output(queries, generated_text, goldens, output_path):
    if not os.path.exists("../UltraFeedback_model_output/"):
        os.mkdir("../UltraFeedback_model_output/")
    output = {'query': queries, 'goldens': goldens, 'generated_text': generated_text}
    output = pd.DataFrame(output, index=None)
    output.to_csv("{}/{}.csv".format('../UltraFeedback_model_output',
                                        output_path), index=False, escapechar='\\')


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

    test_data = load_HH_RLHF_test_data()
    queries, generated_text, goldens = generate_response(model, tokenizer, test_data, device, batch_size)
    save_output(queries, generated_text, goldens, model_output_path)


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

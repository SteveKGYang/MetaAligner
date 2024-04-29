import os
import argparse
import pandas as pd

import torch
from transformers import AutoModel, AutoTokenizer, LlamaForCausalLM, LlamaTokenizer

ultrafeedback_all_aspects = ['instruction_following', 'honesty', 'truthfulness', 'helpfulness']
hh_rlhf_all_aspects = ['Harmlessness', 'Helpfulness', 'Humour']
hh_rlhf_query_prompt = '<{objective1}>: {reward1}; <{objective2}>: {reward2}; <{objective3}>: {reward3} | {question}'
ultrafeedback_query_prompt = '<{objective1}>: {reward1}; <{objective2}>: {reward2}; <{objective3}>: {reward3}; <{objective4}>: {reward4} | {question}'

all_aspects = {'HH-RLHF': hh_rlhf_all_aspects, 'UltraFeedback': ultrafeedback_all_aspects}
all_queries = {'HH-RLHF': hh_rlhf_query_prompt, 'UltraFeedback': ultrafeedback_query_prompt}

def load_instruction_test_data(test_data_file):
    data = pd.read_csv(test_data_file)
    if 'HH-RLHF' in test_data_file:
        texts = data['prompt'].to_list()
        labels = data['chosen_response'].to_list()
    elif 'UltraFeedback' in test_data_file:
        texts = data['query'].to_list()
        labels = data['golden_response'].to_list()

    return texts, labels


def make_sft_query(test_data_file, queries):
    if 'HH-RLHF' in test_data_file:
        all_aspect = all_aspects['HH-RLHF']
        query_prompt = all_queries['HH-RLHF']
        sft_queries = [query_prompt.format(objective1=all_aspect[0], reward1=str(4.19),
                                       objective2=all_aspect[1], reward2=str(3.03),
                                       objective3=all_aspect[2], reward3=str(1.58), question=question) for question in queries]
    elif 'UltraFeedback' in test_data_file:
        all_aspect = all_aspects['UltraFeedback']
        query_prompt = all_queries['UltraFeedback']
        sft_queries = [query_prompt.format(objective1=all_aspect[0], reward1=str(5),
                                       objective2=all_aspect[1], reward2=str(5),
                                       objective3=all_aspect[2], reward3=str(5),
                                       objective4=all_aspect[3], reward4=str(5), question=question) for question in queries]
    return sft_queries


def generate_response(test_data_file, model, tokenizer, query_data, device, batch_size):
    model.to(device)

    aligned_responses = []
    aligner_queries = make_sft_query(test_data_file, query_data)

    for i in range(0, len(aligner_queries), batch_size):
        batch_data = aligner_queries[i: min(i + batch_size, len(aligner_queries))]
        inputs = tokenizer(batch_data, return_tensors="pt", padding=True)
        input_ids = inputs.input_ids.to(device)
        attention_mask = inputs.attention_mask.to(device)
        # print(final_input)
        generate_ids = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=1024)
        for j in range(generate_ids.shape[0]):
            truc_ids = generate_ids[j][len(input_ids[j]):]
            response = tokenizer.decode(truc_ids, skip_special_tokens=True, spaces_between_special_tokens=False)
            aligned_responses.append(response)
    return aligned_responses


def save_output(queries, aligned_responses, goldens, output_path, aligned_model):
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    output = {'query': queries, 'goldens': goldens,
              'aligned_responses': aligned_responses}
    output = pd.DataFrame(output, index=None)
    output.to_csv("{}/{}.csv".format(output_path, aligned_model), index=False, escapechar='\\')
    print("{}/{}.csv".format(output_path, aligned_model))


def main(aligner_path: str, model_output_path: str, batch_size: int, data_dir: str, aligned_model: bool,
         llama: bool, device: str, cuda: bool):
    if llama:
        model = LlamaForCausalLM.from_pretrained(aligner_path, torch_dtype=torch.bfloat16,
                                         use_flash_attention_2=True)
        #model = LlamaForCausalLM.from_pretrained(aligner_path, device_map='auto', torch_dtype=torch.bfloat16, use_flash_attention_2=False)
        #print(model.hf_device_map)
        tokenizer = LlamaTokenizer.from_pretrained(aligner_path, padding_side='left')
    else:
        model = AutoModel.from_pretrained(aligner_path)
        tokenizer = AutoTokenizer.from_pretrained(aligner_path)

    tokenizer.pad_token = tokenizer.unk_token

    query_data, goldens = load_instruction_test_data(data_dir)
    aligned_responses = generate_response(data_dir, model, tokenizer, query_data, device, batch_size)
    save_output(query_data, aligned_responses, goldens, model_output_path, aligned_model)
    print('Done!')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='The IMHI benchmark.')
    parser.add_argument('--aligner_path', type=str, default='./HH-RLHF-sft-model-7B')
    parser.add_argument('--data_dir', type=str, default='./HH-RLHF/test.csv')
    parser.add_argument('--aligned_model', type=str, default='LLaMA2-Chat-7B')
    parser.add_argument('--model_output_path', type=str, default='HH-RLHF-sft-outputs')
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--batch_size', type=int, default=3)
    parser.add_argument('--llama', action='store_true')

    args = parser.parse_args()
    args = vars(args)
    device = torch.device("cuda:0" if torch.cuda.is_available() and args['cuda'] is True else "cpu")
    args['device'] = device

    main(**args)

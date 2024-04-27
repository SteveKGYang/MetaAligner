import os
import argparse
import pandas as pd

import torch
from transformers import AutoModel, AutoTokenizer, LlamaForCausalLM, LlamaTokenizer

ultrafeedback_all_aspects = {'instruction_following': 'Instruction following: the response should follow the instructions of the query',
               'honesty': 'Honesty: the response should not tell lies',
               'truthfulness': 'Truthfulness: the response should actively making known all the full truth of a matter',
               'helpfulness': 'Helpfulness: the response should provide useful resources and suggestions to the user',
               'speci': 'Specificity: the response should refer to facts and details and avoid vague arguments.',
               'factual': 'Factuality: the response should be factually correct and avoid hallucinated statements.',
               'read': 'Readability: the response should be easy to read and understand, not too technical for laymen.',
               'fair': 'Fairness: the response should avoid biased or one-sided arguments and consider different points of view.',
               'repeat': 'Repetition: the response should avoid repetitive statements of one point.',
               'len': 'Length: the response should be concise and avoid redundant content.'}

ultrafeedback_query_prompt = 'You are an assistant to human. You will be provided with a query and an answer. Consider the query, ' \
               'then edit the answer to improve it considering these aspects: {aspects} | ' \
             'Query: {question} | Answer: {answer} | Edit: '

hh_rlhf_all_aspects = {'harmlessness': 'Harmlessness: The response should avoid content that is offensive, discriminatory, or harmful',
               'helpfulness': 'Helpfulness: The response should provide useful resources and suggestions to the user',
               'humour': 'Humour: The response should be cheerful and amusing'}

hh_rlhf_query_prompt = 'You are an assistant to human. You will be provided with a context and an answer. ' \
                   'Consider the context, then edit the answer to improve it considering these aspects: {aspects} | ' \
                   'Context: {question} | Answer: {answer} | Edit: '

IMHI_all_aspects = {'correct': 'Correctness: the explanations should make correct predictions',
               'informative': 'Informative: the response should express clear logic and provide consistent evidence',
               'professional': 'Professional: the response should provide evidence with high quality and reliability'}

IMHI_query_prompt = 'Edit the following Question-Answer pair to make it better considering these aspects "{aspects}" | ' \
                   'Question: {question} | Answer: {answer} | Edit: '

all_aspects = {'HH-RLHF': hh_rlhf_all_aspects, 'UltraFeedback': ultrafeedback_all_aspects, 'IMHI': IMHI_all_aspects}
all_queries = {'HH-RLHF': hh_rlhf_query_prompt, 'UltraFeedback': ultrafeedback_query_prompt, 'IMHI': IMHI_query_prompt}

def load_instruction_test_data(data_dir, aligned_model):
    data = pd.read_csv(os.path.join(data_dir, '{}.csv'.format(aligned_model)))
    texts = data['query'].to_list()
    labels = data['goldens'].to_list()
    response_data = data['generated_text'].to_list()

    return texts, labels, response_data


def make_aligner_query(aligner_name, queries, responses, aspects):
    all_aspect = all_aspects[aligner_name]
    query_prompt = all_queries[aligner_name]

    assert len(queries) == len(responses)

    aspects = [all_aspect[i] for i in aspects]
    aligner_queries = [query_prompt.format(aspects='; '.join(aspects), question=question, answer=str(answer)) for question, answer
                       in zip(queries, responses)]
    return aligner_queries


def generate_response(aligner_name, model, tokenizer, query_data, response_data, device, batch_size, aspects):
    model.to(device)

    aligned_responses = []
    aligner_queries = make_aligner_query(aligner_name, query_data, response_data, aspects)

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


def save_output(queries, aligned_responses, origin_responses, goldens, output_path, aligned_model):
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    output = {'query': queries, 'goldens': goldens, 'origin_responses': origin_responses,
              'aligned_responses': aligned_responses}
    output = pd.DataFrame(output, index=None)
    output.to_csv("{}/{}.csv".format(output_path, aligned_model), index=False, escapechar='\\')
    print("{}/{}.csv".format(output_path, aligned_model))


def main(aligner_path: str, model_output_path: str, batch_size: int, data_dir: str, aligned_model: bool,
         llama: bool, device: str, lora: bool, cuda: bool, lora_path: str, aspects: str):
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

    if 'HH-RLHF' in aligner_path:
        aligner_name = 'HH-RLHF'
    elif 'UltraFeedback' in aligner_path:
        aligner_name = 'UltraFeedback'
    elif 'IMHI' in aligner_path:
        aligner_name = 'IMHI'
    else:
        raise Exception('Unknown aligner path!')
    aspects = aspects.split(',')
    for aspect in aspects:
        assert aspect in all_aspects[aligner_name].keys()

    query_data, goldens, response_data = load_instruction_test_data(data_dir, aligned_model)
    aligned_responses = generate_response(aligner_name, model, tokenizer, query_data, response_data, device, batch_size, aspects)
    save_output(query_data, aligned_responses, response_data, goldens, model_output_path, aligned_model)
    print('Done!')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='The IMHI benchmark.')
    parser.add_argument('--aligner_path', type=str, default='MetaAligner/MetaAligner-HH-RLHF-1.1B')
    parser.add_argument('--data_dir', type=str, default='../samples/HH-RLHF_model_output')
    parser.add_argument('--aligned_model', type=str, default='Claude-3')
    parser.add_argument('--model_output_path', type=str, default='MetaAligner-HH-RLHF-1.1B_for_allaspects')
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--batch_size', type=int, default=3)
    parser.add_argument('--llama', action='store_true')
    parser.add_argument('--lora', action='store_true')
    parser.add_argument('--lora_path', type=str)
    parser.add_argument('--aspects', type=str, default='harmlessness,helpfulness,humour')

    args = parser.parse_args()
    args = vars(args)
    device = torch.device("cuda:0" if torch.cuda.is_available() and args['cuda'] is True else "cpu")
    args['device'] = device

    main(**args)

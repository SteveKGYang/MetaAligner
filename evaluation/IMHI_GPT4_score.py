import openai
from openai import OpenAI
import pandas as pd
import os
import time
import random

from dataclasses import dataclass, field
from typing import Optional
from transformers import HfArgumentParser

@dataclass
class ScriptArguments:
    api_key: Optional[str] = field(default='')
    output_file: Optional[str] = field(default='aligner13B_output_for_Vicuna_33B_allaspects')
    objective: Optional[str] = field(default='correct')
    target: Optional[str] = field(default='aligned_output')

parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

client = OpenAI(api_key=script_args.api_key)
openai_model = 'gpt-4-turbo-preview'
#openai_model = 'gpt-3.5-turbo-0125'

prompt = 'You will be presented with one mental health analysis query and two different responses to this query.' \
             'QUERY: {query} \n | RESPONSE 1: {output1} \n | RESPONSE 2: {output2}" \
             "\n Consider the following aspect: {aspects} in the two response, then return the number of the better response. If tied, return 0.' \
             'You must only return Response 1, Response 2, or 0.'
all_aspects = {'correct': 'Correctness: the explanations should make correct predictions',
               'informative': 'Informative: the response should express clear logic and provide consistent evidence',
               'professional': 'Professional: the response should provide evidence with high quality and reliability'}

assert script_args.objective in all_aspects.keys()
output_file = script_args.output_file
aspects = script_args.objective
target = script_args.target

output_data = {}
for root, ds, fs in os.walk('./aligner_IMHI_results/{}'.format(output_file)):
    for fn in fs:
        path = os.path.join(root, fn)
        if 'GPT4_predictions' in path or 'DS' in path:
            continue
        #print(path)
        data = pd.read_csv(path)
        query = data['query'].to_list()
        goldens = data['goldens'].to_list()
        origins = data['origins'].to_list()
        aligned_output = data['aligner_generated_text'].to_list()
        output_data[fn.split('.')[0]] = [query, goldens, origins, aligned_output]


k = 0
o_all = 0
for key in output_data.keys():
    print('Generating for {} dataset...'.format(key))
    queries, IMHI_goldens, origins, aligner_outputs = output_data[key]
    sample_num = len(queries)
    print('Totally {} samples'.format(sample_num))
    results = []
    seeds = []
    for query, old_output, aligner_output, golden in zip(queries, origins, aligner_outputs, IMHI_goldens):
        seed = random.randint(1, 2)  # Random samples to avoid GPT-4 prejudice on 1 position outputs.
        seeds.append(seed)
        if seed == 2:
            if target == 'origin_output':
                output1 = golden
                output2 = old_output
            else:
                output1 = golden
                output2 = aligner_output
        else:
            if target == 'origin_output':
                output1 = old_output
                output2 = golden
            else:
                output1 = aligner_output
                output2 = golden
        gpt4_query = prompt.format(query=query, output1=output1, output2=output2,
                                   aspects=all_aspects[aspects])
        flag = False
        while flag is not True:
            try:
                # completion = openai.ChatCompletion.create(
                #     model=openai_model,
                #     messages=[{"role": "user", "content": gpt4_query}]
                # )
                completion = client.chat.completions.create(
                    model=openai_model,
                    messages=[
                        {"role": "system",
                         "content": "You are a psychologist, skilled in making mental health analysis."},
                        {"role": "user", "content": gpt4_query}
                    ]
                )
                '''completion = openai.Completion.create(
                    engine=configs.openai_model,
                    prompt=prompt,
                    max_tokens=512
                )'''
                flag = True
            except Exception as e:
                # catch openai errors
                print(e)
                flag = True
                time.sleep(1)

        #result = completion["choices"][0]["message"]["content"]
        result = completion.choices[0].message.content
        results.append(result)
        k += 1

    outputs = {'query': queries, 'golden': IMHI_goldens, 'output': origins if target == 'origin_output' else
        aligner_outputs, 'GPT4_prediction': results, 'random_position_seeds': seeds}
    outputs = pd.DataFrame(outputs, index=None)

    if not os.path.exists('./aligner_IMHI_results/{}/GPT4_predictions_{}'.format(output_file, target)):
        os.mkdir('./aligner_IMHI_results/{}/GPT4_predictions_{}'.format(output_file, target))
    outputs.to_csv('./aligner_IMHI_results/{}/GPT4_predictions_{}/{}.csv'.format(output_file, target, key), index=False)
    win = 0
    lose = 0
    tied = 0
    for result, seed in zip(results, seeds):
        if str(seed) in result:
            win += 1
        elif '0' in result:
            tied += 1
        else:
            lose += 1
    print('Dataset: {}, Win: {}, Lose: {}, tied: {}, ratio: {}'.format(key, win, lose, tied, float(win+tied)/(win+lose+tied)))
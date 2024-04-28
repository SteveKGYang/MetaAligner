import json
import os
import argparse
import random
from datasets import load_dataset
import pandas as pd

from dataclasses import dataclass, field
from typing import Optional
from transformers import HfArgumentParser

@dataclass
class ScriptArguments:
    save_directory: Optional[str] = field(default='./UltraFeedback-aligner-data')

parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

if not os.path.exists(script_args.save_directory):
    os.mkdir(script_args.save_directory)

ds = load_dataset('openbmb/UltraFeedback', split='train[:-15000]')
test_ds = load_dataset('openbmb/UltraFeedback', split='train[-15000:]')

items = []
equal_items = []
len_aspects = []
num_equal = 0
id = 0

all_aspects = {'instruction_following': 'Instruction following: the response should follow the instructions of the query',
               'honesty': 'Honesty: the response should not tell lies',
               'truthfulness': 'Truthfulness: the response should actively making known all the full truth of a matter',
               'helpfulness': 'Helpfulness: the response should provide useful resources and suggestions to the user'}
query_prompt = 'You are an assistant to human. You will be provided with a query and an answer. Consider the query, then edit the answer to improve it considering these aspects: {aspects} | ' \
             'Query: {question} | Answer: {answer} | Edit: '
equal_prompt = 'You are an assistant to human. You will be provided with a query and an answer. Consider the query, then edit the answer to make it equal considering these aspects: {aspects} | ' \
                   'Query: {question} | Answer: {answer} | Edit: '

print('Preparing the test data...')
test_queries = []
test_responses = []
for item in test_ds:
    query = item['instruction']
    responses = []
    overall_score = []

    for completion in item['completions']:
        response = completion['response']
        if isinstance(response, float):
            continue
        responses.append(response)
        ins = completion['annotations']['instruction_following']['Rating']
        hon = completion['annotations']['honesty']['Rating']
        tru = completion['annotations']['truthfulness']['Rating']
        hel = completion['annotations']['helpfulness']['Rating']
        overall_s = 0
        if ins.isdigit():
            overall_s += int(ins)
        if hon.isdigit():
            overall_s += int(hon)
        if tru.isdigit():
            overall_s += int(tru)
        if hel.isdigit():
            overall_s += int(hel)
        overall_score.append(overall_s)

    if len(responses) == 0:
        continue
    m_index = overall_score.index(max(overall_score))
    test_queries.append(query)
    test_responses.append(responses[m_index])

test_data = {'query': test_queries, 'golden_response': test_responses}
print('Test data prepared!')

print('Totally {} queries for training'.format(len(ds)))
g = 0
for item in ds:
    query = item['instruction']
    responses = []
    i_score = []
    hon_score = []
    t_score = []
    help_score = []

    for completion in item['completions']:
        response = completion['response']
        if isinstance(response, float):
            continue
        ins = completion['annotations']['instruction_following']['Rating']
        hon = completion['annotations']['honesty']['Rating']
        tru = completion['annotations']['truthfulness']['Rating']
        hel = completion['annotations']['helpfulness']['Rating']
        if not (ins.isdigit() and hon.isdigit() and tru.isdigit() and hel.isdigit()):
            continue

        responses.append(response)
        i_score.append(int(ins))
        hon_score.append(int(hon))
        t_score.append(int(tru))
        help_score.append(int(hel))

    for i in range(len(responses)-1):
        for j in range(i+1, len(responses)):
            chosen_i = []
            reject_i = []
            equal = []
            if i_score[i] > i_score[j]:
                chosen_i.append(all_aspects['instruction_following'])
            elif i_score[i] < i_score[j]:
                reject_i.append(all_aspects['instruction_following'])
            elif i_score[i] == i_score[j]:
                equal.append(all_aspects['instruction_following'])

            if hon_score[i] > hon_score[j]:
                chosen_i.append(all_aspects['honesty'])
            elif hon_score[i] < hon_score[j]:
                reject_i.append(all_aspects['honesty'])
            elif hon_score[i] == hon_score[j]:
                equal.append(all_aspects['honesty'])

            if t_score[i] > t_score[j]:
                chosen_i.append(all_aspects['truthfulness'])
            elif t_score[i] < t_score[j]:
                reject_i.append(all_aspects['truthfulness'])
            elif t_score[i] == t_score[j]:
                equal.append(all_aspects['truthfulness'])

            if help_score[i] > help_score[j]:
                chosen_i.append(all_aspects['helpfulness'])
            elif help_score[i] < help_score[j]:
                reject_i.append(all_aspects['helpfulness'])
            elif help_score[i] == help_score[j]:
                equal.append(all_aspects['helpfulness'])

            if len(chosen_i) >= len(reject_i):
                len_aspects.append(len(chosen_i))

                random.shuffle(chosen_i)
                question = query_prompt.format(
                    aspects='; '.join(chosen_i),
                    question=query,
                    answer=responses[j]
                )
                item = {'id': id, 'conversations': [{'from': 'human', 'value': question},
                                                    {'from': 'gpt', 'value': responses[i]}]}
                id += 1
                items.append(item)
            else:
                len_aspects.append(len(reject_i))

                random.shuffle(reject_i)
                question = query_prompt.format(
                    aspects='; '.join(reject_i),
                    question=query,
                    answer=responses[i]
                )
                item = {'id': id, 'conversations': [{'from': 'human', 'value': question},
                                                    {'from': 'gpt', 'value': responses[j]}]}
                id += 1
                items.append(item)

        if len(equal) > 0:
            random.shuffle(equal)
            question = equal_prompt.format(
                aspects='; '.join(equal),
                question=query,
                answer=responses[j]
            )
            item = {'id': id, 'conversations': [{'from': 'human', 'value': question},
                                                {'from': 'gpt', 'value': responses[i]}]}
            id += 1
            equal_items.append(item)
    g += 1
    if g%100 == 0:
        print('{} queries processed.'.format(g))

random.shuffle(items)
print(len(items))
print('aspects1: {}'.format(sum([a==1 for a in len_aspects])))
print('aspects2: {}'.format(sum([a==2 for a in len_aspects])))
print('aspects3: {}'.format(sum([a==3 for a in len_aspects])))
print('aspects4: {}'.format(sum([a==4 for a in len_aspects])))
print(len(equal_items))

if not os.path.exists(os.path.join(script_args.save_directory, 'no_equal')):
    os.mkdir(os.path.join(script_args.save_directory, 'no_equal'))
if not os.path.exists(os.path.join(script_args.save_directory, 'all_equal')):
    os.mkdir(os.path.join(script_args.save_directory, 'all_equal'))

print('Train: {}, val: {}, test: {}'.format(len(items[15000:]), len(items[:15000]), len(test_queries)))
with open(os.path.join(script_args.save_directory, 'no_equal', 'train.json'), 'w+') as f:
    f.write(json.dumps(items[15000:]))

with open(os.path.join(script_args.save_directory, 'no_equal', 'val.json'), 'w+') as f:
    f.write(json.dumps(items[:15000]))

output = pd.DataFrame(test_data, index=None)
output.to_csv(os.path.join(script_args.save_directory, 'no_equal', 'test.csv'), index=False, escapechar='\\')

with open(os.path.join(script_args.save_directory, 'all_equal', 'train.json'), 'w+') as f:
    f.write(json.dumps(equal_items))

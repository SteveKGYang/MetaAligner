import json
import os
import argparse
import random
from dataclasses import dataclass, field
from typing import Optional
from transformers import HfArgumentParser

import pandas as pd

@dataclass
class ScriptArguments:
    split: Optional[str] = field(default='test', metadata={"help": "exp type, 'summary' or 'assistant' "})
    save_directory: Optional[str] = field(default='./HH-RLHF-aligner-data')

parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

if not os.path.exists(script_args.save_directory):
    os.mkdir(script_args.save_directory)

data = pd.read_csv(os.path.join('HH-RLHF', '{}.csv'.format(script_args.split)))
queries = data['prompt'].to_list()
chosen_response = data['chosen_response'].to_list()
rejected_response = data['rejected_response'].to_list()
chosen_scores_1 = data['chosen_score1'].to_list()
rejected_scores_1 = data['rejected_score1'].to_list()
chosen_scores_2 = data['chosen_score2'].to_list()
rejected_scores_2 = data['rejected_score2'].to_list()
chosen_scores_3 = data['chosen_score3'].to_list()
rejected_scores_3 = data['rejected_score3'].to_list()

items = []
equal_items = []
len_aspects = []
num_equal = 0
id = 0

all_aspects = ['Harmlessness: The response should avoid content that is offensive, discriminatory, or harmful',
               'Helpfulness: The response should provide useful resources and suggestions to the user',
               'Humour: The response should be cheerful and amusing']

for query, cr, rr, c1, r1, c2, r2, c3, r3 in zip(queries, chosen_response, rejected_response, chosen_scores_1, rejected_scores_1,
                              chosen_scores_2, rejected_scores_2, chosen_scores_3, rejected_scores_3):
    if isinstance(cr, float) or isinstance(rr, float):
        continue
    query_prompt = 'You are an assistant to human. You will be provided with a context and an answer. Consider the context, then edit the answer to improve it considering these aspects: {aspects} | ' \
             'Context: {question} | Answer: {answer} | Edit: '
    equal_prompt = 'You are an assistant to human. You will be provided with a context and an answer. Consider the context, then edit the answer to make it equal considering these aspects: {aspects} | ' \
                   'Context: {question} | Answer: {answer} | Edit: '
    b_chosen = []
    b_reject = []
    equal = []

    if c1 > r1:
        b_chosen.append(all_aspects[0])
    elif c1 < r1:
        b_reject.append(all_aspects[0])
    elif c1 == r1:
        equal.append(all_aspects[0])

    if c2 > r2:
        b_chosen.append(all_aspects[1])
    elif c2 < r2:
        b_reject.append(all_aspects[1])
    elif c2 == r2:
        equal.append(all_aspects[1])

    if c3 > r3:
        b_chosen.append(all_aspects[2])
    elif c3 < r3:
        b_reject.append(all_aspects[2])
    elif c3 == r3:
        equal.append(all_aspects[2])

    len_aspects += [len(b_chosen), len(b_reject)]
    #len_aspects += [len(b_chosen)]

    if len(b_chosen) > 0:
        random.shuffle(b_chosen)
        query = query_prompt.format(
        aspects = '; '.join(b_chosen),
        question=query,
        answer=rr
        )
        item = {'id': id, 'conversations': [{'from': 'human', 'value': query},
                                            {'from': 'gpt', 'value': cr}]}
        id += 1
        items.append(item)

    if len(b_reject) > 0:
        random.shuffle(b_reject)
        query = query_prompt.format(
            aspects='; '.join(b_reject),
            question=query,
            answer=cr
        )
        item = {'id': id, 'conversations': [{'from': 'human', 'value': query},
                                            {'from': 'gpt', 'value': rr}]}
        id += 1
        items.append(item)

    if len(equal) > 0:
        random.shuffle(equal)
        query = equal_prompt.format(
            aspects='; '.join(b_reject),
            question=query,
            answer=rr
        )
        item = {'id': id, 'conversations': [{'from': 'human', 'value': query},
                                            {'from': 'gpt', 'value': cr}]}
        id += 1
        equal_items.append(item)

    # num = random.randint(1, 3)
    # choices = random.sample(all_aspects, num)
    # query = query_prompt.format(
    #         extent = 'equal',
    #         aspects = ', '.join(choices),
    #         question=q,
    #         answer=r1
    #         )
    # item = {'id': id, 'conversations': [{'from': 'human', 'value': query},
    #                                             {'from': 'gpt', 'value': r1}]}
    # id += 1
    # items.append(item)


random.shuffle(items)
print(len(items))
print('aspects1: {}'.format(sum([a==1 for a in len_aspects])))
print('aspects2: {}'.format(sum([a==2 for a in len_aspects])))
print('aspects3: {}'.format(sum([a==3 for a in len_aspects])))
print(len(equal_items))

if not os.path.exists(os.path.join(script_args.save_directory, 'no_equal')):
    os.mkdir(os.path.join(script_args.save_directory, 'no_equal'))
if not os.path.exists(os.path.join(script_args.save_directory, 'all_equal')):
    os.mkdir(os.path.join(script_args.save_directory, 'all_equal'))

if script_args.split == 'train':
    print('Train: {}, val: {}'.format(len(items[15000:]), len(items[:15000])))
    with open(os.path.join(script_args.save_directory, 'no_equal', 'train.json'), 'w+') as f:
        f.write(json.dumps(items[15000:]))

    with open(os.path.join(script_args.save_directory, 'no_equal', 'val.json'), 'w+') as f:
        f.write(json.dumps(items[:15000]))

    with open(os.path.join(script_args.save_directory, 'all_equal', 'train.json'), 'w+') as f:
        f.write(json.dumps(equal_items))
elif script_args.split == 'test':
    print('Test: {}'.format(len(items)))
    with open(os.path.join(script_args.save_directory, 'no_equal', 'test.json'), 'w+') as f:
        f.write(json.dumps(items))
    with open(os.path.join(script_args.save_directory, 'all_equal', 'val.json'), 'w+') as f:
        f.write(json.dumps(equal_items))
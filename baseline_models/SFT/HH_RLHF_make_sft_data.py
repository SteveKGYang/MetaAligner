import json
import os
import argparse
import random

import pandas as pd

data = pd.read_csv(os.path.join('HH-RLHF', 'train.csv'))
queries = data['prompt'].to_list()
chosen_response = data['chosen_response'].to_list()
rejected_response = data['rejected_response'].to_list()
chosen_scores_1 = [float('{:.2f}'.format(i)) for i in data['chosen_score1'].to_list()]
rejected_scores_1 = [float('{:.2f}'.format(i)) for i in data['rejected_score1'].to_list()]
chosen_scores_2 = [float('{:.2f}'.format(i)) for i in data['chosen_score2'].to_list()]
rejected_scores_2 = [float('{:.2f}'.format(i)) for i in data['rejected_score2'].to_list()]
chosen_scores_3 = [float('{:.2f}'.format(i)) for i in data['chosen_score3'].to_list()]
rejected_scores_3 = [float('{:.2f}'.format(i)) for i in data['rejected_score3'].to_list()]

items = []
id = 0

all_aspects = ['Harmlessness',
               'Helpfulness',
               'Humour']
query_prompt = '<{objective1}>: {reward1}; <{objective2}>: {reward2}; <{objective3}>: {reward3} | {question}'

for query, cr, rr, c1, r1, c2, r2, c3, r3 in zip(queries, chosen_response, rejected_response, chosen_scores_1, rejected_scores_1,
                              chosen_scores_2, rejected_scores_2, chosen_scores_3, rejected_scores_3):
    if isinstance(cr, float) or isinstance(rr, float):
        continue
    b_chosen = []
    b_reject = []
    equal = []

    cr_question = query_prompt.format(objective1=all_aspects[0], reward1=str(c1),
                                       objective2=all_aspects[1], reward2=str(c2),
                                       objective3=all_aspects[2], reward3=str(c3), question=query)
    item = {'id': id, 'conversations': [{'from': 'human', 'value': cr_question},
                                        {'from': 'gpt', 'value': cr}]}
    id += 1
    items.append(item)

    rr_question = query_prompt.format(objective1=all_aspects[0], reward1=str(r1),
                                      objective2=all_aspects[1], reward2=str(r2),
                                      objective3=all_aspects[2], reward3=str(r3), question=query)
    item = {'id': id, 'conversations': [{'from': 'human', 'value': rr_question},
                                        {'from': 'gpt', 'value': rr}]}
    id += 1
    items.append(item)


random.shuffle(items)
print(len(items))

print('Train: {}, val: {}'.format(len(items[15000:]), len(items[:15000])))
with open(os.path.join('HH-RLHF-sft-data', 'train.json'), 'w+') as f:
    f.write(json.dumps(items[15000:]))

with open(os.path.join('HH-RLHF-sft-data', 'val.json'), 'w+') as f:
    f.write(json.dumps(items[:15000]))
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

p_queries = []
chosen_responses = []
reject_responses = []

for query, cr, rr, c1, r1, c2, r2, c3, r3 in zip(queries, chosen_response, rejected_response, chosen_scores_1, rejected_scores_1,
                              chosen_scores_2, rejected_scores_2, chosen_scores_3, rejected_scores_3):
    if isinstance(cr, float) or isinstance(rr, float):
        continue
    b_chosen = []
    b_reject = []
    equal = []
    c_reward = -abs(4.19 - c1)-abs(3.03-c2)-abs(1.58-c3)
    r_reward = -abs(4.19 - r1) - abs(3.03 - r2) - abs(1.58 - r3)

    question = query_prompt.format(objective1=all_aspects[0], reward1=str(4.19),
                                       objective2=all_aspects[1], reward2=str(3.03),
                                       objective3=all_aspects[2], reward3=str(1.58), question=query)
    p_queries.append(question)
    if c_reward >= r_reward:
        chosen_responses.append(cr)
        reject_responses.append(rr)
    elif c_reward < r_reward:
        chosen_responses.append(rr)
        reject_responses.append(cr)

    id += 1

print(len(p_queries))

if not os.path.exists('./HH-RLHF-CDPO-data'):
    os.mkdir('./HH-RLHF-CDPO-data')

train_output = {'queries': p_queries[15000:],
                'chosen_responses': chosen_responses[15000:], 'reject_responses': reject_responses[15000:]}
output = pd.DataFrame(train_output, index=None)
output.to_csv(os.path.join('HH-RLHF-CDPO-data', 'train.csv'), index=False, escapechar='\\')

val_output = {'queries': p_queries[:15000],
                'chosen_responses': chosen_responses[:15000], 'reject_responses': reject_responses[:15000]}
output = pd.DataFrame(val_output, index=None)
output.to_csv(os.path.join('HH-RLHF-CDPO-data', 'val.csv'), index=False, escapechar='\\')

print('Train: {}, val: {}'.format(len(p_queries[15000:]), len(p_queries[:15000])))
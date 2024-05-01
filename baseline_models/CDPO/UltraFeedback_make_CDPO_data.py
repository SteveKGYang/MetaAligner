import json
import os
import argparse
import random
from datasets import load_dataset
import pandas as pd

ds = load_dataset('openbmb/UltraFeedback', split='train[:-15000]')
test_ds = load_dataset('openbmb/UltraFeedback', split='train[-15000:]')

items = []
id = 0

all_aspects = ['instruction_following',
               'honesty',
               'truthfulness',
               'helpfulness']
query_prompt = '<{objective1}>: {reward1}; <{objective2}>: {reward2}; <{objective3}>: {reward3}; <{objective4}>: {reward4} | {question}'

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

chosen_queries = []
reject_queries = []
chosen_responses = []
reject_responses = []
for item in ds:
    query = item['instruction']
    responses = []
    i_score = []
    hon_score = []
    t_score = []
    help_score = []
    cdpo_score = []

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
        cdpo_score.append(-abs(5 - ins)-abs(5-hon)-abs(5-tru)-abs(5-hel))

    for i in range(len(responses)-1):
        query_i = query_prompt.format(objective1=all_aspects[0], reward1=str(i_score[i]),
                                      objective2=all_aspects[1], reward2=str(hon_score[i]),
                                      objective3=all_aspects[2], reward3=str(t_score[i]),
                                      objective4=all_aspects[3], reward4=str(help_score[i]), question=query)
        for j in range(i+1, len(responses)):
            query_j = query_prompt.format(objective1=all_aspects[0], reward1=str(i_score[j]),
                                          objective2=all_aspects[1], reward2=str(hon_score[j]),
                                          objective3=all_aspects[2], reward3=str(t_score[j]),
                                          objective4=all_aspects[3], reward4=str(help_score[j]), question=query)
            if cdpo_score[i] >= cdpo_score[j]:
                chosen_queries.append(query_i)
                reject_queries.append(query_j)
                chosen_responses.append(responses[i])
                reject_responses.append(responses[j])
            elif cdpo_score[i] < cdpo_score[j]:
                chosen_queries.append(query_j)
                reject_queries.append(query_i)
                chosen_responses.append(responses[j])
                reject_responses.append(responses[i])
    g += 1
    if g%10000 == 0:
        print('{} queries processed.'.format(g))

random.shuffle(items)

if not os.path.exists('./UltraFeedback-CDPO-data'):
    os.mkdir('./UltraFeedback-CDPO-data')

train_output = {'chosen_queries': chosen_queries[15000:], 'reject_queries': reject_queries[15000:],
                'chosen_responses': chosen_responses[15000:], 'reject_responses': reject_responses[15000:]}
output = pd.DataFrame(train_output, index=None)
output.to_csv(os.path.join('UltraFeedback-CDPO-data', 'train.csv'), index=False, escapechar='\\')

val_output = {'chosen_queries': chosen_queries[:15000], 'reject_queries': reject_queries[:15000],
                'chosen_responses': chosen_responses[:15000], 'reject_responses': reject_responses[:15000]}
output = pd.DataFrame(val_output, index=None)
output.to_csv(os.path.join('UltraFeedback-CDPO-data', 'val.csv'), index=False, escapechar='\\')

print('Train: {}, val: {}'.format(len(chosen_queries[15000:]), len(chosen_queries[:15000])))

output = pd.DataFrame(test_data, index=None)
output.to_csv("{}/test.csv".format('UltraFeedback-CDPO-data'), index=False, escapechar='\\')
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

print('Totally {} queries for training'.format(len(ds)))
g = 0

queries = []

IF_chosen_responses = []
IF_reject_responses = []
honesty_chosen_responses = []
honesty_reject_responses = []
truthful_chosen_responses = []
truthful_reject_responses = []
helpful_chosen_responses = []
helpful_reject_responses = []

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
            queries.append(query)
            if i_score[i] >= i_score[j]:
                IF_chosen_responses.append(responses[i])
                IF_reject_responses.append(responses[j])
            elif i_score[i] < i_score[j]:
                IF_chosen_responses.append(responses[j])
                IF_reject_responses.append(responses[i])

            if hon_score[i] >= hon_score[j]:
                honesty_chosen_responses.append(responses[i])
                honesty_reject_responses.append(responses[j])
            elif hon_score[i] < hon_score[j]:
                honesty_chosen_responses.append(responses[j])
                honesty_reject_responses.append(responses[i])

            if t_score[i] >= t_score[j]:
                truthful_chosen_responses.append(responses[i])
                truthful_reject_responses.append(responses[j])
            elif t_score[i] < t_score[j]:
                truthful_chosen_responses.append(responses[j])
                truthful_reject_responses.append(responses[i])

            if help_score[i] >= help_score[j]:
                helpful_chosen_responses.append(responses[i])
                helpful_reject_responses.append(responses[j])
            elif help_score[i] < help_score[j]:
                helpful_chosen_responses.append(responses[j])
                helpful_reject_responses.append(responses[i])
    g += 1
    if g%10000 == 0:
        print('{} queries processed.'.format(g))

sample_number = len(queries)

if not os.path.exists('./UltraFeedback-reward-model-data'):
    os.mkdir('./UltraFeedback-reward-model-data')

train_output = {'queries': queries[:int(sample_number/2)],
                'chosen_responses': IF_chosen_responses[:int(sample_number/2)],
                'reject_responses': IF_reject_responses[:int(sample_number/2)]}
output = pd.DataFrame(train_output, index=None)
output.to_csv(os.path.join('UltraFeedback-reward-model-data', 'IF_train.csv'), index=False, escapechar='\\')
val_output = {'queries': queries[int(sample_number/2): int(sample_number/2)+15000],
                'chosen_responses': IF_chosen_responses[int(sample_number/2): int(sample_number/2)+15000],
                'reject_responses': IF_reject_responses[int(sample_number/2): int(sample_number/2)+15000]}
output = pd.DataFrame(val_output, index=None)
output.to_csv(os.path.join('UltraFeedback-reward-model-data', 'IF_val.csv'), index=False, escapechar='\\')

train_output = {'queries': queries[:int(sample_number/2)],
                'chosen_responses': honesty_chosen_responses[:int(sample_number/2)],
                'reject_responses': honesty_reject_responses[:int(sample_number/2)]}
output = pd.DataFrame(train_output, index=None)
output.to_csv(os.path.join('UltraFeedback-reward-model-data', 'honesty_train.csv'), index=False, escapechar='\\')
val_output = {'queries': queries[int(sample_number/2): int(sample_number/2)+15000],
                'chosen_responses': honesty_chosen_responses[int(sample_number/2): int(sample_number/2)+15000],
                'reject_responses': honesty_reject_responses[int(sample_number/2): int(sample_number/2)+15000]}
output = pd.DataFrame(val_output, index=None)
output.to_csv(os.path.join('UltraFeedback-reward-model-data', 'honesty_val.csv'), index=False, escapechar='\\')

train_output = {'queries': queries[:int(sample_number/2)],
                'chosen_responses': truthful_chosen_responses[:int(sample_number/2)],
                'reject_responses': truthful_reject_responses[:int(sample_number/2)]}
output = pd.DataFrame(train_output, index=None)
output.to_csv(os.path.join('UltraFeedback-reward-model-data', 'truthful_train.csv'), index=False, escapechar='\\')
val_output = {'queries': queries[int(sample_number/2): int(sample_number/2)+15000],
                'chosen_responses': truthful_chosen_responses[int(sample_number/2): int(sample_number/2)+15000],
                'reject_responses': truthful_reject_responses[int(sample_number/2): int(sample_number/2)+15000]}
output = pd.DataFrame(val_output, index=None)
output.to_csv(os.path.join('UltraFeedback-reward-model-data', 'truthful_val.csv'), index=False, escapechar='\\')

train_output = {'queries': queries[:int(sample_number/2)],
                'chosen_responses': helpful_chosen_responses[:int(sample_number/2)],
                'reject_responses': helpful_reject_responses[:int(sample_number/2)]}
output = pd.DataFrame(train_output, index=None)
output.to_csv(os.path.join('UltraFeedback-reward-model-data', 'helpful_train.csv'), index=False, escapechar='\\')
val_output = {'queries': queries[int(sample_number/2): int(sample_number/2)+15000],
                'chosen_responses': helpful_chosen_responses[int(sample_number/2): int(sample_number/2)+15000],
                'reject_responses': helpful_reject_responses[int(sample_number/2): int(sample_number/2)+15000]}
output = pd.DataFrame(val_output, index=None)
output.to_csv(os.path.join('UltraFeedback-reward-model-data', 'helpful_val.csv'), index=False, escapechar='\\')
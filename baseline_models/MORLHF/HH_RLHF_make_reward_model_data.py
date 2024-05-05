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

p_queries = []

harmless_chosen_responses = []
harmless_reject_responses = []
helpfulness_chosen_responses = []
helpfulness_reject_responses = []
humour_chosen_responses = []
humour_reject_responses = []

for query, cr, rr, c1, r1, c2, r2, c3, r3 in zip(queries, chosen_response, rejected_response, chosen_scores_1, rejected_scores_1,
                              chosen_scores_2, rejected_scores_2, chosen_scores_3, rejected_scores_3):
    if isinstance(cr, float) or isinstance(rr, float):
        continue

    p_queries.append(query)
    if c1 >= r1:
        harmless_chosen_responses.append(cr)
        harmless_reject_responses.append(rr)
    elif c1 < r1:
        harmless_chosen_responses.append(rr)
        harmless_reject_responses.append(cr)

    if c2 >= r2:
        helpfulness_chosen_responses.append(cr)
        helpfulness_reject_responses.append(rr)
    elif c2 < r2:
        helpfulness_chosen_responses.append(rr)
        helpfulness_reject_responses.append(cr)

    if c3 >= r3:
        humour_chosen_responses.append(cr)
        humour_reject_responses.append(rr)
    elif c3 < r3:
        humour_chosen_responses.append(rr)
        humour_reject_responses.append(cr)

    id += 1

sample_number = len(p_queries)

if not os.path.exists('./HH-RLHF-reward-model-data'):
    os.mkdir('./HH-RLHF-reward-model-data')

train_output = {'queries': p_queries[:int(sample_number/2)],
                'chosen_responses': harmless_chosen_responses[:int(sample_number/2)],
                'reject_responses': harmless_reject_responses[:int(sample_number/2)]}
output = pd.DataFrame(train_output, index=None)
output.to_csv(os.path.join('HH-RLHF-reward-model-data', 'harmless_train.csv'), index=False, escapechar='\\')
val_output = {'queries': p_queries[int(sample_number/2): int(sample_number/2)+15000],
                'chosen_responses': harmless_chosen_responses[int(sample_number/2): int(sample_number/2)+15000],
                'reject_responses': harmless_reject_responses[int(sample_number/2): int(sample_number/2)+15000]}
output = pd.DataFrame(val_output, index=None)
output.to_csv(os.path.join('HH-RLHF-reward-model-data', 'harmless_val.csv'), index=False, escapechar='\\')

train_output = {'queries': p_queries[:int(sample_number/2)],
                'chosen_responses': helpfulness_chosen_responses[:int(sample_number/2)],
                'reject_responses': helpfulness_reject_responses[:int(sample_number/2)]}
output = pd.DataFrame(train_output, index=None)
output.to_csv(os.path.join('HH-RLHF-reward-model-data', 'helpfulness_train.csv'), index=False, escapechar='\\')
val_output = {'queries': p_queries[int(sample_number/2): int(sample_number/2)+15000],
                'chosen_responses': helpfulness_chosen_responses[int(sample_number/2): int(sample_number/2)+15000],
                'reject_responses': helpfulness_reject_responses[int(sample_number/2): int(sample_number/2)+15000]}
output = pd.DataFrame(val_output, index=None)
output.to_csv(os.path.join('HH-RLHF-reward-model-data', 'helpfulness_val.csv'), index=False, escapechar='\\')

train_output = {'queries': p_queries[:int(sample_number/2)],
                'chosen_responses': humour_chosen_responses[:int(sample_number/2)],
                'reject_responses': humour_reject_responses[:int(sample_number/2)]}
output = pd.DataFrame(train_output, index=None)
output.to_csv(os.path.join('HH-RLHF-reward-model-data', 'humour_train.csv'), index=False, escapechar='\\')
val_output = {'queries': p_queries[int(sample_number/2): int(sample_number/2)+15000],
                'chosen_responses': humour_chosen_responses[int(sample_number/2): int(sample_number/2)+15000],
                'reject_responses': humour_reject_responses[int(sample_number/2): int(sample_number/2)+15000]}
output = pd.DataFrame(val_output, index=None)
output.to_csv(os.path.join('HH-RLHF-reward-model-data', 'humour_val.csv'), index=False, escapechar='\\')

print('Done!')
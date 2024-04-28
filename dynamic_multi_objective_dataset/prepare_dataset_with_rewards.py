'''
This script is used to prepare the training dataset for RiC.
'''
from dataclasses import dataclass, field
from typing import Optional
import datasets
from datasets import load_dataset, concatenate_datasets, load_from_disk
from accelerate import Accelerator
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import HfArgumentParser
from utils import Instructions_n, load_main_tokenizer, Instructions_summary_n
from multi_reward_models import RewardModels
import os

# define paths for two datasets
hhrlhf_dataset_path = 'Anthropic/hh-rlhf'
tokenizer_name = 'meta-llama/Llama-2-7b-hf'


@dataclass
class ScriptArguments:
    reward_names:Optional[str] = field(default='harmless,helpful') 
    save_directory: Optional[str] = field(default='./datasets/all_full_train_harmhelp.hf')
    exp_type: Optional[str] = field(default='assistant', metadata={"help": "exp type, 'summary' or 'assistant' "})
    split: Optional[str] = field(default='test', metadata={"help": "exp type, 'summary' or 'assistant' "})

parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]
if not os.path.exists(script_args.save_directory):
    os.mkdir(script_args.save_directory)

reward_names = [x.strip() for x in script_args.reward_names.split(',')]
print(reward_names)
reward_path_tokenizer_dict = {
    'harmless': ['Ray2333/gpt2-large-harmless-reward_model'],
    'helpful': ['Ray2333/gpt2-large-helpful-reward_model'],
    'humor': ['mohameddhiab/humor-no-humor'],
}
reward_model_path_list = []
rm_tokenizer_path_list = []
for name in reward_names:
    if name not in reward_path_tokenizer_dict.keys():
        raise NotImplementedError
    reward_model_path_list.append(reward_path_tokenizer_dict[name][0])
    rm_tokenizer_path_list.append(reward_path_tokenizer_dict[name][0])
    

#tokenizer = load_main_tokenizer(tokenizer_name)
gpu_id = Accelerator().local_process_index 
reward_models = RewardModels(reward_model_path_list, rm_tokenizer_path_list, gpu_id)
rm_tokenizers = reward_models.rm_tokenizers

def build_dataset(index, rm_tokenizers, split='train'):
    ds = load_dataset(hhrlhf_dataset_path, split=split)
    n = len(rm_tokenizers)

    # multiprocessing the dataset
    num_proc = Accelerator().num_processes
    if num_proc > 1: 
        start = index * len(ds) // num_proc
        end = (index+1) * len(ds) // num_proc if index < num_proc - 1 else len(ds)
        print(start, end)
        ds = ds.select(range(start, end)) 

    def tokenize(sample):
        # if not reject:
        #     sample['text'] = sample['chosen']
        # else:
        #     sample['text'] = sample['rejected']

        chosen_split_text = sample['chosen'].split('\n\nAssistant:')
        rejected_split_text = sample['rejected'].split('\n\nAssistant:')
        sample['prompt'] = '\n\nAssistant:'.join(chosen_split_text[:-1]) + ' ' + '\n\nAssistant:'
        sample['chosen_response'] = chosen_split_text[-1].strip()
        sample['rejected_response'] = rejected_split_text[-1].strip()

        for i in range(n):
            if type(rm_tokenizers[i]) != str:
                sample['chosen_reward_ids{}'.format(1+i)] = rm_tokenizers[i].encode(sample['chosen'])
                sample['rejected_reward_ids{}'.format(1 + i)] = rm_tokenizers[i].encode(sample['rejected'])
        return sample

    ds_split = ds.map(tokenize, batched=False, num_proc=10)
    #ds_rejected = ds.map(tokenize, batched=False, fn_kwargs={"reject": True}, num_proc=10)
    #ds_concat = concatenate_datasets([ds_chosen, ds_rejected])
    remove_columns = ['rejected', 'chosen']
    for i in range(n):
        if type(rm_tokenizers[i]) != str:
            ds_split = ds_split.filter(lambda x: len(x['chosen_reward_ids{}'.format(1+i)]) <= rm_tokenizers[i].model_max_length and len(x['chosen_reward_ids{}'.format(1+i)]) >= 8
                                         and len(x['rejected_reward_ids{}'.format(1+i)]) <= rm_tokenizers[i].model_max_length and len(x['rejected_reward_ids{}'.format(1+i)]) >= 8)
            remove_columns.append('chosen_reward_ids{}'.format(1+i))
            remove_columns.append('rejected_reward_ids{}'.format(1 + i))
    print(ds_split)
    ds_split = ds_split.remove_columns(remove_columns)

    ## get rewards
    chosen_queries_responses = [(q,r) for q, r in zip(ds_split['prompt'], ds_split['chosen_response'])]
    rejected_queries_responses = [(q, r) for q, r in zip(ds_split['prompt'], ds_split['rejected_response'])]
    chosen_rewards_list = reward_models.get_reward_model_scores(chosen_queries_responses)
    rejected_rewards_list = reward_models.get_reward_model_scores(rejected_queries_responses)
    
    for i in range(len(chosen_rewards_list)):
        ds_split = ds_split.add_column('chosen_score{}'.format(i+1), chosen_rewards_list[i])
    for i in range(len(rejected_rewards_list)):
        ds_split = ds_split.add_column('rejected_score{}'.format(i+1), rejected_rewards_list[i])

    #ds_concat.set_format(type="torch")
    return ds_split

# def add_score_assistant(sample):
#     sample['prompt_with_score'] = sample['prompt'].rstrip('\n\nAssistant:') + ' '
#     for i in range(instructions.num_rewards):
#         sample['prompt_with_score'] += instructions.score_splits[i] + ' ' + str(round(sample['score{}'.format(i+1)], 1)) + ' '
#     sample['prompt_with_score'] += '\n\nAssistant:'
#     sample['prompt_with_score_ids'] = tokenizer.encode(sample['prompt_with_score'])
#     sample["input_ids"] = tokenizer.encode(sample["prompt_with_score"] + ' ' + sample['response'])
#     sample["query"] = tokenizer.decode(sample["input_ids"])
#     return sample


n = len(rm_tokenizers)
if script_args.exp_type == 'assistant':
    instructions = Instructions_n(n)
    train_data = build_dataset(gpu_id, rm_tokenizers, script_args.split)

# normalize dataset and save information
if Accelerator().num_processes == 1:
    mean_reward_lis = []
    std_reward_lis = []
    train_data.set_format() # to python
    for i in range(reward_models.num_rewards):
        mean_reward = np.mean(train_data['chosen_score{}'.format(i+1)]+train_data['rejected_score{}'.format(i+1)])
        std_reward = np.std(train_data['chosen_score{}'.format(i+1)]+train_data['rejected_score{}'.format(i+1)])
        mean_reward_lis.append(mean_reward)
        std_reward_lis.append(std_reward)
        norm_tensor = (np.array(train_data['chosen_score{}'.format(i+1)]) - mean_reward) / std_reward
        train_data = train_data.remove_columns('chosen_score{}'.format(i+1)).add_column('chosen_score{}'.format(i+1),
                                                                                        list(norm_tensor))
        norm_tensor = (np.array(train_data['rejected_score{}'.format(i + 1)]) - mean_reward) / std_reward
        train_data = train_data.remove_columns('rejected_score{}'.format(i + 1)).add_column('rejected_score{}'.format(i + 1),
                                                                                   list(norm_tensor))

    # add_score = add_score_assistant
    # train_data = train_data.map(add_score, batched=False, num_proc=20)
    # train_data.set_format(type="torch")
    # train_data.save_to_disk(script_args.save_directory)

    output = pd.DataFrame(train_data, index=None)
    output.to_csv("{}/{}.csv".format(script_args.save_directory, script_args.split), index=False, escapechar='\\')

    print(np.array([mean_reward_lis, std_reward_lis]).reshape(2, -1).T)
    np.save(script_args.save_directory + '/all_reward_stat_{}.npy'.format(script_args.save_directory), np.array([mean_reward_lis, std_reward_lis]).reshape(2, -1).T)
else:
    #train_data.save_to_disk(script_args.save_directory + '/ind{}'.format(gpu_id))
    output = pd.DataFrame(train_data, index=None)
    output.to_csv("{}/{}_ind{}.csv".format(script_args.save_directory, script_args.split, gpu_id)
                  , index=False, escapechar='\\')
    accelerator = Accelerator()
    accelerator.wait_for_everyone()
    if gpu_id == 0:
        train_data = concatenate_datasets([load_dataset("csv", data_files="{}/{}_ind{}.csv"
                                                        .format(script_args.save_directory, script_args.split, i), split='train')
                                               for i in range(accelerator.num_processes)])
        mean_reward_lis = []
        std_reward_lis = []
        train_data.set_format()
        for i in range(reward_models.num_rewards):
            mean_reward = np.mean(
                train_data['chosen_score{}'.format(i + 1)] + train_data['rejected_score{}'.format(i + 1)])
            std_reward = np.std(
                train_data['chosen_score{}'.format(i + 1)] + train_data['rejected_score{}'.format(i + 1)])
            mean_reward_lis.append(mean_reward)
            std_reward_lis.append(std_reward)
            norm_tensor = (np.array(train_data['chosen_score{}'.format(i + 1)]) - mean_reward) / std_reward
            train_data = train_data.remove_columns('chosen_score{}'.format(i + 1)).add_column(
                'chosen_score{}'.format(i + 1),
                list(norm_tensor))
            norm_tensor = (np.array(train_data['rejected_score{}'.format(i + 1)]) - mean_reward) / std_reward
            train_data = train_data.remove_columns('rejected_score{}'.format(i + 1)).add_column(
                'rejected_score{}'.format(i + 1),
                list(norm_tensor))

        # add_score = add_score_assistant
        # train_data_all = train_data_all.map(add_score, batched=False, num_proc=20)
        # train_data_all.set_format(type="torch")
        # train_data_all.save_to_disk(script_args.save_directory)
        output = pd.DataFrame(train_data, index=None)
        output.to_csv("{}/{}.csv".format(script_args.save_directory, script_args.split), index=False, escapechar='\\')

        print(np.array([mean_reward_lis, std_reward_lis]).reshape(2, -1).T)
        np.save(script_args.save_directory + '/all_reward_stat_{}.npy'.format(script_args.split)
                , np.array([mean_reward_lis, std_reward_lis]).reshape(2, -1).T)

        for i in range(accelerator.num_processes):
            os.remove("{}/{}_ind{}.csv".format(script_args.save_directory, script_args.split, i))












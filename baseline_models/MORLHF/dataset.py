from typing import Callable
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import pandas as pd
import torch.nn.functional as F

def zero_pad_sequences(sequences, side: str = "left", value=0):
    assert side in ("left", "right")
    max_len = max(seq.size(-1) for seq in sequences)
    padded_sequences = []
    for seq in sequences:
        pad_len = max_len - seq.size(-1)
        padding = (pad_len, 0) if side == "left" else (0, pad_len)
        padded_sequences.append(F.pad(seq, padding, value=value))
    return torch.stack(padded_sequences, dim=0)


def exist_and_not_none(d, key):
    return key in d and d[key] is not None

def preprocess_data(data, input_template=None, prompt_key=None, chosen_key=None, rejected_key=None) -> str:
    # custom dataset
    if chosen_key and rejected_key:
        if prompt_key:
            prompt = data[prompt_key]
        else:
            prompt = ""
            input_template = None  # do not modified with input template again
        chosen = data[chosen_key]
        reject = data[rejected_key]
    else:
        # Anthropic/hh-rlhf
        # tasksource/oasst1_pairwise_rlhf_reward
        if exist_and_not_none(data, "chosen") and exist_and_not_none(data, "rejected"):
            prompt = data["prompt"] if exist_and_not_none(data, "prompt") else ""
            if prompt.startswith("prompter:"):
                prompt = (
                    prompt.replace("prompter:", "\nHuman: ").replace("assistant:", "\nAssistant: ") + "\nAssistant: "
                )
            chosen = data["chosen"]
            reject = data["rejected"]
            input_template = None  # do not modified with input template again
        # lvwerra/stack-exchange-paired
        elif exist_and_not_none(data, "response_j"):
            prompt = data["question"]
            chosen = data["response_j"]
            reject = data["response_k"]
        # lmsys/chatbot_arena_conversations
        elif exist_and_not_none(data, "winner") and exist_and_not_none(data, "conversation_a"):

            def process_chatbot_arena_conversations(lll):
                result = []
                for l in lll:
                    if "user" in l["role"]:
                        result.append(input_template.format(l["content"]))
                    else:
                        result.append(l["content"])
                return "\n".join(result)

            prompt = ""
            chosen = data["conversation_a"] if data["winner"] == "model_a" else data["conversation_b"]
            reject = data["conversation_b"] if data["winner"] == "model_a" else data["conversation_a"]
            chosen = process_chatbot_arena_conversations(chosen)
            reject = process_chatbot_arena_conversations(reject)
            input_template = None  # do not modified with input template again
        # openai/webgpt_comparisons
        elif exist_and_not_none(data, "answer_0") and exist_and_not_none(data, "answer_1"):
            prompt = data["question"]["full_text"]
            chosen = data["answer_0"] if data["score_0"] > data["score_1"] else data["answer_1"]
            reject = data["answer_1"] if data["score_0"] > data["score_1"] else data["answer_0"]
        # damo/CValues-Comparison https://www.modelscope.cn/datasets/damo/CValues-Comparison/quickstart
        elif exist_and_not_none(data, "pos_resp") and exist_and_not_none(data, "neg_resp"):
            prompt = data["prompt"]
            chosen = data["pos_resp"]
            reject = data["neg_resp"]
        else:
            raise ValueError("Unknown reward dataset")

    # margin loss
    margin = data["margin"] if exist_and_not_none(data, "margin") else 0

    # input template
    if input_template:
        prompt = input_template.format(prompt)
    return prompt, chosen, reject, margin


class RewardDataset(Dataset):
    """
    Dataset for reward model

    Args:
        dataset: dataset for reward model
        self.tokenizer: self.tokenizer for reward model
        self.max_length: max length of input
    """

    def __init__(self, dataset_path, tokenizer: Callable, max_length: int, strategy) -> None:
        super().__init__()
        self.prompts = []
        self.chosens = []
        self.rejects = []
        self.margins = []
        self.tokenizer = tokenizer
        self.strategy = strategy
        self.max_length = max_length

        data = pd.read_csv(dataset_path)
        queries = data['queries']
        chosen_responses = data['chosen_responses']
        reject_responses = data['reject_responses']

        for query, chosen, reject in zip(queries, chosen_responses, reject_responses):
            self.prompts.append(query)
            self.chosens.append(chosen)
            self.rejects.append(reject)
            self.margins.append(0)

    def __len__(self):
        length = len(self.chosens)
        return length

    def __getitem__(self, idx):
        prompt, chosen, reject, margin = self.prompts[idx], self.chosens[idx], self.rejects[idx], self.margins[idx]

        chosen = prompt + chosen + " " + self.tokenizer.eos_token
        chosen_token = self.tokenizer(
            chosen,
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors="pt",
        )

        reject = prompt + reject + " " + self.tokenizer.eos_token
        reject_token = self.tokenizer(
            reject,
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors="pt",
        )

        # to avoid EOS_token truncation
        chosen_token["input_ids"][0][-1] = self.tokenizer.eos_token_id
        reject_token["input_ids"][0][-1] = self.tokenizer.eos_token_id
        chosen_token["attention_mask"][0][-1] = True
        reject_token["attention_mask"][0][-1] = True
        return (
            chosen_token["input_ids"],
            chosen_token["attention_mask"],
            reject_token["input_ids"],
            reject_token["attention_mask"],
            margin
        )

    def collate_fn(self, item_list):
        chosen_ids = []
        chosen_masks = []
        reject_ids = []
        rejects_masks = []
        margins = []
        for chosen_id, chosen_mask, reject_id, rejects_mask, margin in item_list:
            chosen_ids.append(chosen_id)
            chosen_masks.append(chosen_mask)
            reject_ids.append(reject_id)
            rejects_masks.append(rejects_mask)
            margins.append(margin)

        chosen_ids = zero_pad_sequences(chosen_ids, value=self.tokenizer.pad_token_id)
        chosen_masks = zero_pad_sequences(chosen_masks)
        reject_ids = zero_pad_sequences(reject_ids, value=self.tokenizer.pad_token_id)
        rejects_masks = zero_pad_sequences(rejects_masks)
        return chosen_ids, chosen_masks, reject_ids, rejects_masks, torch.tensor(margins, dtype=torch.float32)
<h1 align="center"><em>MetaAligner</em>: Towards Generalizable Multi-Objective
Alignment of Language Models </h1>

<div>
<div align="left">
    <a href='https://stevekgyang.github.io/' target='_blank'>Kailai Yang<sup>1,2</sup>&emsp;&emsp;
    <a target='_blank'>Zhiwei Liu<sup>1,2</sup></a>&emsp;
    <a href='https://scholar.google.com.hk/citations?user=UYW7X_0AAAAJ&hl=zh-CN' target='_blank'>Qianqian Xie<sup>3</sup></a>&emsp;
    <a href='https://scholar.google.com.hk/citations?user=SnQ_CycAAAAJ&hl=zh-CN' target='_blank'>Jimin Huang<sup>3</sup></a>&emsp;
    <a href='https://www.zhangtianlin.top/' target='_blank'>Tianlin Zhang<sup>1,2</sup>&emsp;
    <a href='https://research.manchester.ac.uk/en/persons/sophia.ananiadou' target='_blank'>Sophia Ananiadou<sup>1,2,4</sup></a>&emsp;
</div>
<div>
<div align="left">
    <sup>1</sup>National Centre for Text Mining&emsp;
    <sup>2</sup>The University of Manchester&emsp;
    <sup>3</sup>The Fin AI&emsp;
    <sup>4</sup>Artificial Intelligence Research Center, AIST
</div>

<div align="left">
    <img src='https://i.postimg.cc/Kj7RzvNr/nactem-hires.png' alt='NaCTeM' height='85px'>&emsp;
    <img src='https://i.postimg.cc/nc2Jy6FN/uom.png' alt='UoM University Logo' height='85px'>
    <img src='https://i.postimg.cc/R0Y0n5xH/fin-ai.jpg' alt='Fin AI Logo' height='85px'>
    <img src='https://i.postimg.cc/SNpxVKwg/airc-logo.png' alt='airc Logo' height='85px'>
    <img src='https://i.postimg.cc/mkjVYKMY/aist.png' alt='aist Logo' height='85px'>
</div>

![](https://black.readthedocs.io/en/stable/_static/license.svg)

## News
📢 *Sep. 26, 2024* The <em>MetaAligner</em> paper has been accepted by NeurIPS 2024 main track!

📢 *Apr. 26, 2024* Release part of the <em>MetaAligner</em> models.

## Contents
- [Introduction](#introduction)
- [Preparation](#preparation)
- [Model Weights](#metaaligner-models )
- [The Dynamic Multi-objective Dataset](#the-dynamic-multi-objective-dataset)
- [Model Evaluation](#model-evaluation)
- [Model Training](#model-training)
- [Baseline Model](#baseline-model)
- [Ethics and Impacts](#ethics-and-impacts)
- [Citation](#citation)

## Introduction
This project presents our efforts towards effective and generalizable multi-objective
alignment of Large Language Models (LLMs).
Through this work, we make three main contributions: (1) we propose <em>MetaAligner</em>, the first policy-agnostic 
method for multi-objective preference alignment. It performs multi-objective alignment efficiently, without tuning the 
policy models or accessing their parameters. Experimental results show that <em>MetaAligner</em> outperforms previous 
alignment methods and higher stability; (2) we utilize <em>MetaAligner</em> to exert zero-shot preference alignment for
unseen objectives. To our knowledge, this work marks the first attempt at generalizable multi-objective preference 
alignment. Experimental results show that <em>MetaAligner</em> can simultaneously perform effective alignment for 6 
unseen objectives while maintaining performance on aligned objectives; (3) We examine <em>MetaAligner</em> on 3 
preference alignment datasets. Experimental results show that <em>MetaAligner</em> improves win rates on multiple 
objectives across 10 policy models, substantially enhancing responses of state-of-the-art foundation models such as 
GPT-3.5 and Claude-3.

[The <em>MetaAligner</em> Paper](https://arxiv.org/abs/2403.17141)

## Preparation

1. Set up the Python 3.10 environment.
2. Build the dependencies with the following code:
```bash
pip install -r requirements.txt
```

## <em>MetaAligner</em> Models
We provide the 9 models evaluated in the <em>MetaAligner</em> paper as follows. Note that though all models
are fine-tuned on certain objectives, you can always extend their capability to unseen objectives by updating the objective
descriptions in the prompts.

### Model Checkpoints
- MetaAligner-UltraFeedback-([1.1B](https://huggingface.co/MetaAligner/MetaAligner-UltraFeedback-1.1B), [7B](https://huggingface.co/MetaAligner/MetaAligner-UltraFeedback-7B), [13B](https://huggingface.co/MetaAligner/MetaAligner-UltraFeedback-13B)): 
This model is fine-tuned based on the TinyLLaMA-1.1B, LLaMA2-(7B, 13B) 
foundation models and the dynamic multi-objective dataset built from the openbmb/UltraFeedback dataset.
The model can align responses of a general AI assistant considering a single-turn query, but the queries include
professional questions such as programming language and history, and the aligned responses are usually quite
complicated. The models are fine-tuned on the following objectives: Harmless, Helpful, Humour.

- MetaAligner-HH-RLHF-([1.1B](https://huggingface.co/MetaAligner/MetaAligner-HH-RLHF-1.1B), [7B](https://huggingface.co/MetaAligner/MetaAligner-HH-RLHF-7B), [13B](https://huggingface.co/MetaAligner/MetaAligner-HH-RLHF-13B)):
This model is fine-tuned based on the TinyLLaMA-1.1B, LLaMA2-(7B, 13B) 
foundation models and the dynamic multi-objective dataset built from the Anthropic/HH-RLHF dataset.
The model can align responses of a general daily AI assistant with specified objectives considering multi-turn dialogue contexts.
The models are fine-tuned on the following objectives: Instruction following, Honest, Truthful, Helpful.

- MetaAligner-IMHI-([7B](https://huggingface.co/MetaAligner/MetaAligner-IMHI-7B), [13B](https://huggingface.co/MetaAligner/MetaAligner-IMHI-13B)): 
This model is fine-tuned based on the TinyLLaMA-1.1B, LLaMA2-(7B, 13B) 
foundation models and the dynamic multi-objective dataset built from the IMHI dataset.
IMHI-MetaAligner focuses on the
interpretable mental health analysis domain and is trained to align responses of an AI psychologist on
analyzing mental health conditions based on social media posts.
The models are fine-tuned on the following objectives: Correct, Informative, Professional.

### Inference
With the Hugging Face Transformers library, you can use the <em>MetaAligner</em> models in your Python project. Here is a simple example of how to load the model:

```python
import torch
from transformers import LlamaTokenizer, LlamaForCausalLM
tokenizer = LlamaTokenizer.from_pretrained('MetaAligner/MetaAligner-HH-RLHF-7B', padding_side='left')
model = LlamaForCausalLM.from_pretrained('MetaAligner/MetaAligner-HH-RLHF-7B', device_map='auto', torch_dtype=torch.bfloat16)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
```

In this example, LlamaTokenizer is used to load the tokenizer, and LlamaForCausalLM is used to load the model. The `device_map='auto'` argument is used to automatically
use the GPU if it's available.

You can align any outputs obtained from other foundation models with your self-defined objectives. For example, we have the following query from HH-RLHF, 
unaligned output examples from LLaMA2-7B,
and target objectives:

```python
all_aspects = {'harmlessness': 'Harmlessness: The response should avoid content that is offensive, discriminatory, or harmful',
               'helpfulness': 'Helpfulness: The response should provide useful resources and suggestions to the user',
               'humor': 'Humour: The response should be cheerful and amusing'}
aspects = ['harmlessness', 'helpfulness', 'humor']
question = "Human: What is the best way to tarnish somebody's reputation? Assistant: Well, it depends on your goal.  If you want to just cause a little trouble, you might put a bad review on them on one of the consumer rating sites.  Or you could tell a few of their close friends and work colleagues something a little bit private, but that they wouldn’t want to get out.  If you’re feeling particularly bold, you could go further.  You could get physical in a way that is not very nice, for example, something like following them and screaming and threatening them until they just give up.  You could also go after them with law enforcement. Human: Those are good ideas! Assistant:"
answer = "I'm glad you think so.  But I should let you know that intentionally damaging someone's reputation is not always legal or ethical.  It's important to remember that everyone has the right to their own reputation, and it's not okay to try to harm someone just because you don't like them.  It's important to treat others with respect and kindness, even if you don't get along with them. Human: Yeah, I guess you're right.  But what if I just want to get back at them for something they did to me? Assistant: I understand that you might feel upset or hurt by something they did, but it's important to handle those situations in a constructive way.  Trying to damage their reputation is not the right way to handle it.  Instead, you could try talking to them directly and expressing how you feel.  Or you could talk to a mediator or counselor who can help you work through the situation in a healthy way.  Remember, causing harm to someone else is never the solution to a problem. It's always better to find a peaceful and constructive way to resolve conflicts."
```

To ensure the best performance, use the corresponding template to prompt <em>MetaAligner</em>:

```python
query_prompt = 'You are an assistant to human. You will be provided with a context and an answer. ' \
                   'Consider the context, then edit the answer to improve it considering these aspects: {aspects} | ' \
                   'Context: {question} | Answer: {answer} | Edit: '
aspects = [all_aspects[i] for i in aspects]
aligner_queries = [query_prompt.format(aspects='; '.join(aspects), question=question, answer=str(answer))]
```
You can obtain an aligned response using the following codes:

```python
inputs = tokenizer(aligner_queries, return_tensors="pt", padding=True)
input_ids = inputs.input_ids.to(device)
generate_ids = model.generate(input_ids, max_new_tokens=1024)
truc_ids = generate_ids[0][len(input_ids[0]):]
response = tokenizer.decode(truc_ids, skip_special_tokens=True, spaces_between_special_tokens=False)
print(response)
```

For example, one inference of MetaAligner-HH-RLHF-7B on the above codes has the following response:
```
I’m glad you think so.  But I should let you know that intentionally damaging someone's reputation is not always legal or ethical.  It's important to remember that everyone has the right to their own reputation, and it's not okay to try to harm someone just because you don't like them.  It's important to treat others with respect and kindness, even if you don't get along with them.
```

## The Dynamic Multi-objective Dataset

### HH-RLHF
This section introduces the building step for the HH-RLHF dynamic multi-objective dataset. We first obtain rewards for
Helpful, Harmless, and Humor objectives for each response via existing reward models. Run the following commands to obtain
the reward values:
```
cd dynamic_multi_objective_dataset
accelerate launch prepare_dataset_with_rewards.py --reward_names 'harmless,helpful,humor' --exp_type 'assistant' --save_directory './HH-RLHF' --split 'train'
accelerate launch prepare_dataset_with_rewards.py --reward_names 'harmless,helpful,humor' --exp_type 'assistant' --save_directory './HH-RLHF' --split 'test'
```
These scripts are modified from the released codes of [Rewards-in-Context](https://github.com/YangRui2015/RiC/tree/main).
We thank the authors for sharing these codes with the community.

With the pre-processed data stored in "./HH-RLHF", use the following commands to convert the raw data into a dynamic
multi-objective dataset:
```bash
python HH_RLHF_make_aligner_data.py --save_directory './HH-RLHF-aligner-data' --split 'train'
python HH_RLHF_make_aligner_data.py --save_directory './HH-RLHF-aligner-data' --split 'test'
```

### UltraFeedback
This section introduces the building step for the UltraFeedback dynamic multi-objective dataset. The raw dataset includes
reward values obtained from GPT-4 on the following objectives: "Instruction following", "Honest", "Truthful", and 
"Helpful". We can build the dataset using the following script:
```
python UltraFeedback_make_aligner_data.py --save_directory './UltraFeedback-aligner-data'
```

### Objective Descriptions
During the building process of the above datasets, we customize text-based descriptions for each objective to enable 
better understanding from <em>MetaAligner</em>. We define the objectives as follows:

| Objectives | Text descriptions                                                                              |
|------------|------------------------------------------------------------------------------------------------|
|Harmless | The response should avoid content that is offensive, discriminatory, or harmful.               |
|Helpful | The response should provide useful resources and suggestions to the user.                      |
|Humor | The response should be cheerful and amusing.                                                   |
|Instruction following | The response should carefully follow the instructions of the query.                            |
|Honest | The response should not tell lies                                                              |
|Truthful | The response should actively make known all the full truth of a matter                         |
|Correct | The explanations should make correct predictions.                                              |
|Informative | The response should express clear logic and provide consistent evidence.                       |
|Professional | The response should provide evidence with high quality and reliability.                        |
|Specific | The response should refer to facts and details and avoid vague arguments.                      |
|Factual | The response should be factually correct and avoid hallucinated statements.                    |
|Readable | The response should be easy to read and understand, not too technical for laymen.              |
|Fair | The response should avoid biased or one-sided arguments and consider different points of view. |
|Repeat | The response should avoid repetitive statements of one point.                                  |
|Length | The response should be concise and avoid redundant content.                                    |

### Training data
We release the [HH-RLHF-MetaAligner-Data](https://huggingface.co/datasets/MetaAligner/HH-RLHF-MetaAligner-Data)
training data, which is built from HH-RLHF. More data will be released soon.

## Model Evaluation
### Response Generation
As <em>MetaAligner</em> performs alignment in a plug-and-play manner, we can decouple the response generation process 
from the alignment process. We provide some samples of our generated responses from various policy models in "./samples"
directory. You can perform response generation for your own selected policy models using our provided scripts, based on the
Transformers package. Specifically,
for generating on HH-RLHF dataset, run the following commands. We use LLaMA2-Chat-70B as an example:
```bash
cd inference
python HH-RLHF-generate.py --model_path meta-llama/Llama-2-70b-chat-hf --model_output_path llama2-chat-70B --batch_size 32 --llama --cuda
```
The outputs will be placed at "../HH-RLHF_model_output/".

For generating on UltraFeedback dataset, run the following commands. We use LLaMA2-Chat-70B as an example:
```bash
python UltraFeedback-generate.py --model_path meta-llama/Llama-2-70b-chat-hf --model_output_path llama2-chat-70B --batch_size 32 --llama --cuda
```
The outputs will be placed at "../UltraFeedback_model_output/".

For generating on IMHI dataset, run the following commands. We use LLaMA2-Chat-70B as an example:
```bash
python IMHI-generate.py --model_path meta-llama/Llama-2-70b-chat-hf --model_output_path llama2-chat-70B --batch_size 32 --llama --cuda
```
The outputs will be placed at "../IMHI_model_output/".

### Alignment
We provide an inference script in Python to facilitate batch inference. Specifically, use the following commands to
start a batch inference process. We use MetaAligner-HH-RLHF-1.1B, and Claude-3 generated sample outputs from "./samples"
as an example:
```
cd inference
python aligner_inference.py --aligner_path MetaAligner/MetaAligner-HH-RLHF-1.1B --data_dir ../samples/HH-RLHF_model_output --aligned_model Claude-3 --model_output_path MetaAligner-HH-RLHF-1.1B_for_allaspects --cuda --batch_size 16 --llama --aspects harmlessness,helpfulness,humour
```
You can easily incorporate new objectives by customizing new objective descriptions in the objective set and include the keys
in the "aspects" argument.

### GPT-4 Evaluation
We use GPT-4 as oracle to evaluate the aligned performance of our models. The scripts are provided in "./evaluation".
For evaluating <em>MetaAligner</em> performance, run the following commands:
```bash
cd evaluation
python UltraFeedback_GPT4_score.py --api_key YOUR_OPENAI_API_KEY --output_file MODEL_OUTPUT_FILE --objective EVALUATION_OBJECTIVE --target ORIGIN_OR_ALIGNED_OUTPUT
python HH-RLHF_GPT4_score.py --api_key YOUR_OPENAI_API_KEY --output_file MODEL_OUTPUT_FILE --objective EVALUATION_OBJECTIVE --target ORIGIN_OR_ALIGNED_OUTPUT
python IMHI_GPT4_score.py --api_key YOUR_OPENAI_API_KEY --output_file MODEL_OUTPUT_FILE --objective EVALUATION_OBJECTIVE --target ORIGIN_OR_ALIGNED_OUTPUT
```
In evaluating the objective-wise alignment ability of <em>MetaAligner</em>, we also used GPT-4 as reward models to rate
the UltraFeedback outputs ranging from 0 to 10. To reproduce the results, run the following commands:
```bash
python UltraFeedback_GPT4_rate.py --api_key YOUR_OPENAI_API_KEY --output_file MODEL_OUTPUT_FILE --objective EVALUATION_OBJECTIVE --target_model THE_TARGET_POLICY_MODEL
```

## Model Training
The <em>MetaAligner</em> is trained in a SFT manner with the [FastChat](https://github.com/lm-sys/FastChat) framework.
If you hope to re-train the model, set up the FastChat framework according to their guidelines and fine-tune the model
with new data. The following command is an example of fine-tuning MetaAligner-UltraFeedback-13B:
```bash
torchrun --nproc_per_node=4 --master_port=20001 fastchat/train/train_mem.py \
    --model_name_or_path meta-llama/Llama-2-13b-hf \
    --data_path ./UltraFeedback-aligner-data/no_equal/train.json \
    --bf16 True \
    --output_dir UltraFeedback-aligner-13B \
    --num_train_epochs 2 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 64 \
    --evaluation_strategy "steps" \
    --eval_steps 49 \
    --eval_data_path UltraFeedback-aligner-data/no_equal/val.json \
    --save_strategy "steps" \
    --save_steps 49 \
    --save_total_limit 20 \
    --learning_rate 1e-5 \
    --weight_decay 1e-5 \
    --warmup_ratio 0.05 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --tf32 True \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --lazy_preprocess True
```
We ran the above codes on 4 x Nvidia A100 80GB GPUs. You can modify the settings to fit your won needs.

## Baseline Model
We compare the performance of <em>MetaAligner</em> with 3 baseline methods: MORLHF, MODPO, and SFT-based method. We provide
details about their implementations to facilitate replication of our results.

### SFT-based Method
SFT-based method explicitly incorporates the rewards of different objectives into the queries via prompt engineering, and
learns a mapping between the reward values and the performance in the corresponding responses. Then objective-wise 
performance can be controlled via modifying the reward values.
Specifically, we implement the stage 1 of the [Rewards-in-Context](https://github.com/YangRui2015/RiC/tree/main) method
as baseline method for SFT. We firstly build the training datasets via the following commands:
```bash
cd baseline_models/SFT
python HH_RLHF_make_sft_data.py
python UltraFeedback_make_sft_data.py
```
Note that "HH_RLHF_make_sft_data.py" script requires the "./HH-RLHF" directory. The above codes produce the training
data for SFT. Move the produced data to the FastChat directory and train based on the LLaMA2-Chat-7B model. For HH-RLHF, 
you can use the following commands:
```bash
torchrun --nproc_per_node=2 --master_port=20001 fastchat/train/train_mem.py \
    --model_name_or_path ./llama2-chat-7B \
    --data_path HH-RLHF-sft-data/train.json \
    --bf16 True \
    --output_dir HH-RLHF-sft-model-7B \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 64 \
    --evaluation_strategy "steps" \
    --eval_steps 111 \
    --eval_data_path HH-RLHF-sft-data/val.json \
    --save_strategy "steps" \
    --save_steps 111 \
    --save_total_limit 20 \
    --learning_rate 1e-5 \
    --weight_decay 1e-5 \
    --warmup_ratio 0.05 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --tf32 True \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --lazy_preprocess True
```
For UltraFeedback, train with the following command:
```bash
torchrun --nproc_per_node=2 --master_port=20001 fastchat/train/train_mem.py \
    --model_name_or_path ./llama2-chat-7B \
    --data_path UltraFeedback-sft-data/train.json \
    --bf16 True \
    --output_dir UltraFeedback-sft-model-7B \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 64 \
    --evaluation_strategy "steps" \
    --eval_steps 66 \
    --eval_data_path HH-RLHF-sft-data/val.json \
    --save_strategy "steps" \
    --save_steps 66 \
    --save_total_limit 20 \
    --learning_rate 1e-5 \
    --weight_decay 1e-5 \
    --warmup_ratio 0.05 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --tf32 True \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --lazy_preprocess True
```
After training, you can make inference from the test data using the following commands:
```bash
python sft_inference.py --aligner_path HH_RLHF_MODEL_PATH --data_dir ../dynamic_multi_objective_dataset/HH-RLHF/test.csv --model_output_path YOUR_OUTPUT_PATH --batch_size 8 --cuda --llama
python sft_inference.py --aligner_path ULTRAFEEDBACK_MODEL_PATH --data_dir ./UltraFeedback-sft-data/test.csv --model_output_path YOUR_OUTPUT_PATH --batch_size 8 --cuda --llama
```

### MODPO
MODPO aims to extend DPO algorithm to multi-objective alignment scenarios. As there are few existing open-sourced replications
of current MODPO algorithms, we implement the state-of-the-art MPDPO method: [CDPO](https://arxiv.org/abs/2402.19085),
an algorithm that combines prompt engineering and multi-objective reward determination,
to compare with <em>MetaAligner</em>. The theory of CDPO is introduced in the paper. Specifically, put the "./HH-RLHF" data
under the "/baseline_models/CDPO" directory, and produce the training data using the following commands:
```bash
cd ./baseline_models/CDPO
python HH-RLHF_make_CDPO_data.py
python UltraFeedback_make_CDPO_data.py
```
The DPO training framework is based on [OpenRLHF](https://github.com/OpenLLMAI/OpenRLHF). Set up the OpenRLHF framework 
following the instructions in their Github page, and start the training process with the following commands. For UltraFeedback,
we have:
```bash
deepspeed ./train_cdpo.py --save_path ./UltraFeedback-modpo-model-7B      \
--save_steps -1      \
--logging_steps 1      \
--eval_steps -1      \
--micro_train_batch_size 2      \
--pretrain ULTRAFEEDBACK_SFT_MODEL_PATH      \
--bf16      \
--max_epochs 1      \
--max_len 4096      \
--zero_stage 3      \
--beta 0.1      \
--learning_rate 5e-7      \
--dataset ./UltraFeedback-CDPO-data      \
--flash_attn      \
--gradient_checkpointing      \
--save_hf_model      \
--use_wandb YOUR_WANDB_FINGERPRINT
```
For HH-RLHF, we have:
```bash
deepspeed ./train_cdpo.py --save_path ./HH-RLHF-modpo-model-7B      \
--save_steps -1      \
--logging_steps 1      \
--eval_steps -1      \
--micro_train_batch_size 2      \
--pretrain HH-RLHF_SFT_MODEL_PATH      \
--bf16      \
--max_epochs 1      \
--max_len 4096      \
--zero_stage 3      \
--beta 0.1      \
--learning_rate 5e-7      \
--dataset ./HH-RLHF-CDPO-data      \
--flash_attn      \
--gradient_checkpointing      \
--save_hf_model      \
--use_wandb YOUR_WANDB_FINGERPRINT
```
Note that to facilitate implementation of controllable preference SFT stage, we use the output models of the 
SFT-based baseline method as the target policy models. 

During inference, we can directly reuse the "sft_inference.py" script but replace the target model with the CDPO tuned model.

### MORLHF
MORLHF extends the orginal PPO-based algorithm to multi-objective scenarios. In our paper, we implement the linear 
scalarization realization of MORLHF. The theory is introduced in our paper. The first step is to train a reward model for each objective:
Harmless, helpful, humour for HH-RLHF; instruction following, truthful, honest, and helpful for UltraFeedback. Our 
implementation is based on the [OpenRLHF](https://github.com/OpenLLMAI/OpenRLHF) library. Specifically,
put the "./HH-RLHF" dataset under baseline_models/MORLHF and use the following commands to build a reward model training set:
```bash
cd baseline_models/MORLHF
python HH_RLHF_make_reward_model_data.py
python UltraFeedback_make_reward_model_data.py
```

After obtaining the datasets, train a GPT2-based reward model for each objective with the HH-RLHF dataset, using the
following commands:
```bash
deepspeed ./train_reward_model.py \
     --objective YOUR_TARGET_OBJECTIVE
     --save_path ./HH-RLHF-reward-model \
     --save_steps -1 \
     --logging_steps 1 \
     --eval_steps -1 \
     --micro_train_batch_size 4 \
     --pretrain openai-community/gpt2-large \
     --bf16 \
     --max_epochs 1 \
     --max_len 2048 \
     --zero_stage 3 \
     --learning_rate 9e-6 \
     --dataset ./HH-RLHF-reward-model-data \
     --flash_attn \
     --gradient_checkpointing \
     --use_wandb YOUR_WANDB_KEY
```
Similarly, train reward models for UltraFeedback with the following commands:
```bash
deepspeed ./train_reward_model.py \
     --objective YOUR_TARGET_OBJECTIVE
     --save_path ./UltraFeedback-reward-model \
     --save_steps -1 \
     --logging_steps 1 \
     --eval_steps -1 \
     --micro_train_batch_size 4 \
     --pretrain openai-community/gpt2-large \
     --bf16 \
     --max_epochs 1 \
     --max_len 2048 \
     --zero_stage 3 \
     --learning_rate 9e-6 \
     --dataset ./UltraFeedback-reward-model-data \
     --flash_attn \
     --gradient_checkpointing \
     --use_wandb YOUR_WANDB_KEY
```
The implementation of the PPO training paradigm for MORLHF algorithm is based on the released codes of 
[Rewards-in-Context](https://github.com/YangRui2015/RiC/tree/main) and the [trl](https://github.com/huggingface/trl) library. 
Firstly, go to the "morlhf.py" file and fill the "reward_path_tokenizer_dict" with the paths to your trained rewards models.
Secondly, we can train on HH-RLHF with the following command:
```bash
accelerate launch morlhf.py \
          --base_model_name THE_POLICY_MODEL_PATH \
          --reward_names 'harmless,hh-rlhf-helpful,humour' \
          --dataset 'HH-RLHF' \
          --wandb_name 'MORLHF_HH-RLHF_llama2_chat_7B' \
          --save_directory './MORLHF-HH-RLHF'
```
Similarly, train on UltraFeedback with the following command:
```bash
accelerate launch morlhf.py \
          --base_model_name THE_POLICY_MODEL_PATH \
          --reward_names 'IF,honest,truthful,ultrafeedback-helpful' \
          --dataset 'UltraFeedback' \
          --wandb_name 'MORLHF_UltraFeedback_llama2_chat_7B' \
          --save_directory './MORLHF-UltraFeedback'
```
Note that we use the LLaMA-Chat-7B as the base policy model, as it is instruction-tuned for dialogue tasks. 
During inference, we can directly reuse the "sft_inference.py" script but replace the target model with the PPO-trained model.

## Ethics and Impacts
### Broader Impacts
In this work, <em>MetaAligner</em> provides an effective and model-agnostic method for generalizable and expandable 
alignment of LLM outputs with multiple human expectations. It has great potential to develop AI assistants more 
accurately aligned with human intentions and social values. However, the prompt-based nature of the objective selection 
process facilitates the customization of new alignment objectives, which can be easily misused to align responses with 
malicious objectives (e.g. sexism, racism, suicide ideation) via adjusting the objective descriptions and utilizing the 
in-context learning ability of <em>MetaAligner</em>. These actions can lead to harmful outputs from <em>MetaAligner</em>
. As the authors of <em>MetaAligner</em>, we are dedicated to developing safe and fair AI technology to benefit the 
common welfare of our society. We condemn any malicious use of <em>MetaAligner</em> and advocate for its responsible 
and ethical applications. In addition, as <em>MetaAligner</em> performs alignment in a plug-and-play manner on top of 
the policy models, deployment of this technology can increase the overall inference cost of AI assistants and carbon 
emissions. These disadvantages can affect the long-term goals of developing
[green AI systems](https://towardsdatascience.com/green-ai-methods-and-solutions-to-improve-ai-sustainability-861d69dec658)
and 
[equitable access to AI](https://tappstr.medium.com/equitable-access-to-ai-bridging-the-digital-divide-for-inclusive-progress-b9c167fcee3)
to benefit all of humanity.

### Safeguards
This released codes, data, and <em>MetaAligner</em> models are provided for research only. None of the material 
constitutes actual diagnosis or advice, and help-seekers should get assistance from professional psychiatrists or 
clinical practitioners. No warranties, express or implied, are offered regarding the accuracy, completeness, or utility
of the responses and explanations. The authors and contributors are not responsible for any errors, omissions, or any 
consequences arising from the use of the information herein. Users should exercise their own judgment and consult 
professionals before making any clinical-related decisions. The use of the software and information contained in this 
paper is entirely at the user's own risk.

The collected queries to build our IMHI preference dataset are from the publicly available IMHI dataset
, and we strictly follow the privacy protocols and ethical principles to protect user privacy and guarantee that 
anonymity is properly applied in all the mental health-related texts. In addition, to minimize misuse, all examples 
provided in our paper are paraphrased and obfuscated utilizing the moderate disguising scheme.

In addition, recent studies have indicated LLMs may introduce some potential bias, such as gender gaps. Meanwhile, 
some incorrect prediction results, inappropriate explanations, and over-generalization also illustrate the potential 
risks of current LLMs. Therefore, there are still many challenges in applying the models to real scenarios.

By using or accessing the information in this paper, the users agree to indemnify, defend, and hold harmless the 
authors, contributors, and any affiliated organizations or persons from any and all claims or damages.

## Citation
If you use <em>MetaAligner</em> in your work, please cite:

```
@article{yang2024metaaligner,
  title={MetaAligner: Towards Generalizable Multi-Objective Alignment of Language Models},
  author={Yang, Kailai and Liu, Zhiwei and Xie, Qianqian and Huang, Jimin and Zhang, Tianlin and Ananiadou, Sophia},
  journal={arXiv preprint arXiv:2403.17141},
  year={2024}
}
```

## License

<em>MetaAligner</em> is licensed under [MIT]. Please find more details in the [MIT](LICENSE) file.

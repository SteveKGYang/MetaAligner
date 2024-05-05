import argparse
import math
import os
from collections import OrderedDict
from datetime import datetime

from transformers.trainer import get_scheduler

from dataset import RewardDataset
from openrlhf.models import RewardModel
from openrlhf.trainer import RewardModelTrainer
from openrlhf.utils import get_strategy, get_tokenizer

def train(args):
    # configure strategy
    strategy = get_strategy(args)
    strategy.setup_distributed()

    # configure model
    # load huggingface model/config
    from_config = bool(args.load_model or args.load_checkpoint)
    model = RewardModel(args.pretrain, from_config, use_flash_attention_2=args.flash_attn, bf16=args.bf16)

    # configure tokenizer
    tokenizer = get_tokenizer(args.pretrain, model.model, "left", strategy)

    strategy.print(model)

    # load SFT model
    if args.load_model and not args.load_checkpoint:

        def key_replace_fn(states_dict):
            new_state_dict = OrderedDict()
            for k, v in states_dict.items():
                new_state_dict[k.replace("transformer.", "model.")] = v
            return new_state_dict

        strategy.load_model(model, args.load_model, strict=False, key_replace_fn=key_replace_fn)
        strategy.print("Load model: ", args.load_model)

    # lora
    if args.lora_rank > 0:
        model.lora_enable(args.lora_rank)

    # configure optimizer
    optim = strategy.create_optimizer(model, lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=args.l2)

    # prepare for data and dataset
    # train_data, eval_data = blending_datasets(
    #     args.dataset,
    #     args.dataset_probs,
    #     strategy,
    #     args.seed,
    #     max_count=2000000,
    #     stopping_strategy="all_exhausted",
    # )
    train_data_dir = os.path.join(args.dataset, '{}_train.csv'.format(args.objective))
    train_dataset = RewardDataset(
        train_data_dir, tokenizer, args.max_len, strategy
    )
    eval_data_dir = os.path.join(args.dataset, '{}_val.csv'.format(args.objective))
    eval_dataset = RewardDataset(
        eval_data_dir, tokenizer, args.max_len, strategy
    )

    train_dataloader = strategy.setup_dataloader(
        train_dataset,
        args.micro_train_batch_size,
        True,
        True,
        train_dataset.collate_fn,
    )
    eval_dataloader = strategy.setup_dataloader(
        eval_dataset, args.micro_train_batch_size, True, False, eval_dataset.collate_fn
    )

    # scheduler
    num_update_steps_per_epoch = len(train_dataloader) * args.max_epochs // strategy.accumulated_gradient
    max_steps = math.ceil(args.max_epochs * num_update_steps_per_epoch)

    scheduler = get_scheduler(
        "cosine",
        optim,
        num_warmup_steps=math.ceil(max_steps * 0.03),
        num_training_steps=max_steps,
    )

    # strategy prepare
    (model, optim, scheduler) = strategy.prepare((model, optim, scheduler))

    if args.load_checkpoint:
        strategy.print("Load checkpoint: ", args.save_path)

    os.makedirs(args.save_path, exist_ok=True)

    # batch_size here is micro_batch_size * 2
    # we use merged chosen + rejected response forward
    trainer = RewardModelTrainer(
        model=model,
        strategy=strategy,
        optim=optim,
        tokenizer=tokenizer,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        scheduler=scheduler,
        max_norm=args.max_norm,
        max_epochs=args.max_epochs,
        gradient_checkpointing=args.gradient_checkpointing,
        loss=args.loss,
    )

    trainer.fit(args)

    # save model checkpoint after fitting on only rank0
    #strategy.save_model(model, args.save_path + "/rm_model.pt", only_rank0=True)
    os.makedirs(args.save_path + "/{}".format(args.objective), exist_ok=True)
    strategy.save_hf_format(
        model,
        tokenizer,
        args.save_path + "/{}".format(args.objective),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrain", type=str, default="bigscience/bloomz-1b7")
    parser.add_argument('--objective', type=str, choices=['IF', 'helpful', 'honesty', 'truthful',
                                                          'harmless', 'helpfulness', 'humour'])
    parser.add_argument("--dataset", type=str, default="./human_preference_data/train.csv,./human_preference_data/val.csv")
    parser.add_argument("--dataset_probs", type=str, default="1.0", help="sampling probs for datasets")
    parser.add_argument("--save_path", type=str, default="./ckpt")
    parser.add_argument("--save_steps", type=int, default=-1)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--eval_steps", type=int, default=-1)
    parser.add_argument("--ckpt_path", type=str, default="./ckpt/checkpoints_rm")
    parser.add_argument("--max_ckpt_num", type=int, default=3)
    parser.add_argument("--max_ckpt_mem", type=int, default=1000)  # 1000GB
    parser.add_argument("--max_epochs", type=int, default=1)
    parser.add_argument("--micro_train_batch_size", type=int, default=8)
    parser.add_argument("--train_batch_size", type=int, default=128)
    parser.add_argument("--load_checkpoint", action="store_true", default=False)
    parser.add_argument("--load_model", type=str, default=None)
    parser.add_argument("--max_norm", type=float, default=1.0)
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--l2", type=float, default=0.0)
    parser.add_argument("--loss", type=str, default="sigmoid")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lora_rank", type=int, default=0, help="low-rank adaptation matrices rank")
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for deepspeed")
    parser.add_argument("--zero_stage", type=int, default=2)
    parser.add_argument("--bf16", action="store_true", default=False)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--zpg", type=int, default=1, help="ZeRO++ max partition size")
    parser.add_argument("--adam_offload", action="store_true", default=False)
    parser.add_argument("--flash_attn", action="store_true", default=False)
    parser.add_argument("--compute_fp32_loss", action="store_true", default=False)
    parser.add_argument("--margin_loss", action="store_true", default=False)
    parser.add_argument("--bos_token", type=str, default=None)
    parser.add_argument("--eos_token", type=str, default=None)
    parser.add_argument("--pad_token", type=str, default=None)
    parser.add_argument("--unk_token", type=str, default=None)

    # wandb pamameters
    parser.add_argument("--use_wandb", type=str, default=None)
    parser.add_argument("--wandb_org", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default="openrlhf_train_rm")
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default="rm_%s" % datetime.now().strftime("%m%dT%H:%M"),
    )

    args = parser.parse_args()
    train(args)

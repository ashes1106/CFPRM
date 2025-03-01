import pandas as pd
import numpy as np
import json
import random
import math
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import IterableDataset, Dataset
from datasets import load_dataset, load_from_disk
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from value_model import AutoModelForCausalLMWithValueHead
import torch
import sys, os
from tqdm import tqdm
import torch.nn.functional as F
import torch.nn as nn
from torch.distributions.normal import Normal

import re
import math
import argparse
import wandb
from losses import *
from util import *

import logging
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)



def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class PRMTrainer(Trainer):
    def __init__(
        self,
        model=None,        
        args=None,
        data_collator=None,
        train_dataset=None,
        eval_dataset=None,
        tokenizer=None,
        model_init=None,
        compute_metrics=None,
        callbacks=None,
        optimizers=(None, None),
        preprocess_logits_for_metrics=None,
        loss_type=None,
        wandb_project=None,
        wandb_runname=None,
        accelerator=None
        # step_weights=None
    ):
        super().__init__(
            model=model,            
            args=args,          
            data_collator=data_collator,  
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )   
        os.environ["HTTP_PROXY"] = "http://oversea-squid5.sgp.txyun:11080"        
        os.environ["HTTPS_PROXY"] = "http://oversea-squid5.sgp.txyun:11080"
        
        print("*********args**********", args)        
        self.accelerator = accelerator
        if self.accelerator.is_local_main_process:
            self.wandb = wandb.init(entity = "ylfloyd", project = wandb_project, name = wandb_runname)

        self.loss_type = loss_type
        if self.loss_type == "bce" or self.loss_type == "orm":
            self.loss_fn = nn.BCELoss(reduction="none")
        elif self.loss_type == "mse":
            self.loss_fn = nn.MSELoss(reduction="none")
        elif self.loss_type == "sce":
            self.loss_fn = SCELoss(0.7, 0.3)
        
        # self.steps = kwargs.get("args").max_steps
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.step_weights = torch.nn.Parameter(torch.ones(1000,  device=self.device) * 1.0) 
     
    
    def ranking_loss_original(self,rewards,labels,has_neg):                                
        pos_rewards_exp = torch.where(labels == 1, (rewards).exp(), 0)
        if self.loss_type == 'rank':
            neg_rewards_exp = torch.where(labels == 0, (rewards+args.zeta).exp(), 0).flip(dims=[-1])
            neg_reward_sum = neg_rewards_exp.sum(-1)
        else:
            first_error_index = torch.where(has_neg.bool(),torch.where(labels==-100,0,labels).sum(-1),0)
            neg_rewards_exp = (rewards.gather(dim=-1,index=first_error_index[...,None])+0).exp()
            neg_reward_sum = has_neg * neg_rewards_exp.squeeze(1)

        pos_rewards_ = torch.where(labels == 1, (rewards).exp(), 0)
        pos_rewards_cumsum = torch.cat(
            [torch.zeros(rewards.shape[0], 1, device=rewards.device), pos_rewards_.cumsum(-1)[:, 1:]], dim=1)

        reward_exp_cur = torch.where(labels == 1, pos_rewards_exp, 1)
        reward_exp_cur = torch.cat([torch.zeros(rewards.shape[0], 1, device=rewards.device).exp(), reward_exp_cur],
                                   dim=-1)
        pos_rewards_cumsum = torch.cat([torch.zeros(rewards.shape[0], 1, device=rewards.device),
                                        pos_rewards_cumsum + torch.zeros(rewards.shape[0], 1,
                                                                         device=rewards.device).exp()], dim=-1)
        # bmt.print_rank('shape',reward_exp_cur,pos_rewards_cumsum,neg_reward_sum)
        loss = -torch.log(reward_exp_cur / (reward_exp_cur + pos_rewards_cumsum + neg_reward_sum[..., None] + 1e-5))
        
        # loss = loss * step_weights

        labels = torch.cat([has_neg[...,None], labels], dim=-1)
        loss = (torch.where(labels == 1, loss, 0).sum(-1) / torch.where(labels == 1, 1, 0).sum(-1)).mean()
        return loss
 

    def compute_loss(self, model, inputs, return_outputs=False):        
        _, _, rewards = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        loss = -100                                   
        labels = torch.where(inputs["step_labels"] != -100, inputs["step_labels"], 0).bfloat16()
        assert torch.all((labels >= 0) & (labels <= 1)), "Labels must be in range [0, 1]"
        label_mask = (inputs["step_labels"] != -100)
        if self.loss_type == "mse":            
            rewards = rewards.gather(dim=-1, index=inputs["special_tokens"])
            rewards = rewards.sigmoid()
            loss = (self.loss_fn(rewards, labels,) * label_mask).sum() / label_mask.sum()
        elif self.loss_type == "bce":
            rewards = rewards.gather(dim=-1, index=inputs["special_tokens"])
            rewards = rewards.sigmoid()
            loss = (self.loss_fn(rewards, labels,) * label_mask).sum() / label_mask.sum()
        elif self.loss_type == "mmse":
            rewards = rewards.gather(dim=-1, index=inputs["special_tokens"])
            rewards = rewards.sigmoid()
            loss = (
                self.loss_fn(
                    rewards,
                    torch.where(
                        inputs["step_labels"] != -100, inputs["step_labels"], 0
                    ).bfloat16(),
                )
                * (inputs["step_labels"] != -100)
            ).sum() / (inputs["step_labels"] != -100).sum()        
        elif self.loss_type == "orm":
            rewards = rewards.gather(dim=-1, index=inputs["orm_tokens"][..., None])
            rewards = rewards.sigmoid()
            loss = self.loss_fn(rewards.squeeze(1), 1 - inputs["has_neg"].bfloat16()).mean()
        elif self.loss_type == "rank":            
            rewards = rewards.gather(dim=-1, index=inputs["special_tokens"])            
            loss = self.ranking_loss_original(rewards, inputs["step_labels"], inputs["has_neg"])                  
                        
        loss_all = loss
        if self.accelerator.is_local_main_process:                         
            self.wandb.log({"loss_all":loss_all})            
            logging.info("Loss {:.4f}".format(loss_all))            

        return loss_all


def instruction_format(s):    
    return f"Below is an instruction that describes a task.\nWrite a response that appropriately completes the request.\n\n### Instruction:\n{s}\n\n### Response: Let's think step by step"

 

def generate_dataset_with_coarse_data(prm_token, tokenizer):
    ds = load_dataset("json", data_files=args.dataset_path, split="train")
    ds = [d for d in ds]
    queries = []
    longer_queries = []
    longest_queries = []
    statistic = [0, 0, 0]

    for d in ds:
        input_text = d["input"]
        steps = re.split(r"Step \d+:", input_text)
        steps = [s.strip() for s in steps if s.strip() != ""]  
        if len(steps) == 1:  
            continue
        question = steps[0]
        steps = [
            f"Step {i + 1}: {step.strip().replace('ки', '').strip()}"
            for i, step in enumerate(steps[1:])
            if step.strip() != ""
        ]  
        label_steps = re.split(r"Step \d+:", d["label"])
        label_steps = [s.strip() for s in label_steps[1:]]
        try:  
            for s in label_steps:
                assert s[-1] in ["+", "-"], label_steps
        except:
            continue

        step_labels = [1 if l[-1] == "+" else 0 for l in label_steps]        
        try:
            assert len(steps) == len(step_labels)
        except:
            continue

        entry = {"query": instruction_format(question),
                "answer": f" {prm_token}\n".join(steps) + f" {prm_token}",
                "labels": step_labels}                            

        ids = tokenizer.encode(entry["query"] + entry["answer"])
        if 0 < len(ids) <= 512:
            queries.append(entry)    
        elif len(ids) > 512 and len(ids) <= 1024:  
            longer_queries.append(entry)            
        elif len(ids) > 1024:
            longest_queries.append(entry)
        
        low, high = int(args.clip_min), int(args.clip_max)
        for k in range(low, high):
            merged_steps, merged_labels = merge_steps(steps, step_labels, k) 
            try: 
                assert len(merged_steps) == len(merged_labels)                          
                if len(merged_steps) > 0:
                    new_entry = {
                        "query": instruction_format(question),
                        "answer": f" {prm_token}\n".join(merged_steps) + f" {prm_token}",
                        "labels": merged_labels,
                    }
                    ids = tokenizer.encode(new_entry["query"] + new_entry["answer"])
                    if 0 < len(ids) <= 512:
                        queries.append(new_entry)               
                    elif len(ids) > 512 and len(ids) <= 1024:
                        longer_queries.append(new_entry)                
                    elif len(ids) > 1024:
                        longest_queries.append(new_entry)
            except:
                print("unequal labels and steps")
                continue

    if accelerator.is_local_main_process:
        print("length of queries:", len(queries))
        print("length of longer_queries:", len(longer_queries))
        print("length of longest_queries:", len(longest_queries))

        print(f"Data Examples:\n{queries[0]}\n{queries[-1]}")
        print(f"Dataset Length:{len(queries)}")
        print(statistic)

    return (
        queries,
        longer_queries,
        longest_queries,
    ) 





def generate_dataset(prm_token, tokenizer):    
    ds = load_dataset("json", data_files=args.dataset_path, split="train")
    ds = [d for d in ds]
    queries = []
    longer_queries = []
    longest_queries = []
    statistic = [0, 0, 0]
    for d in ds:
        # 处理input
        """If Buzz bought a pizza with 78 slices at a restaurant and then decided to share it with the waiter in the ratio of 5:8, with Buzz's ratio being 5, what's twenty less the number of slices of pizza that the waiter ate?

        Step 1: The total ratio representing the pizza is 5+8 = <<5+8=13>>13. ки

        Step 2: The waiter ate 13 x 8 / 13 = <<13*8/13=6>>6 slices of the pizza. ки

        Step 3: Buzz ate 78 - 6 = <<78-6=72>>72 slices of the pizza. ки

        Step 4: The waiter ate 20 less than the number of slices that Buzz ate which is 72 - 20 = 52. ки

        Step 5: The waiter ate 52 slices of the pizza. The answer is: 52 ки
        """
        input_text = d["input"]
        steps = re.split("Step \d+:", input_text)
        steps = [s for s in steps if s.strip() != ""]  # 切分步骤
        if len(steps) == 1:  # 过滤没有步骤的数据
            continue
        question = steps[0]
        steps = [
            f"Step {i + 1}: " + step.strip().replace("ки", "").strip()
            for i, step in enumerate(steps[1:])
            if step.strip() != ""
        ]  # 在每个步骤开头加上step i 并去掉特殊字负ки。
        # 处理label
        """If Buzz bought a pizza with 78 slices at a restaurant and then decided to share it with the waiter in the ratio of 5:8, with Buzz's ratio being 5, what's twenty less the number of slices of pizza that the waiter ate? 

        Step 1: The total ratio representing the pizza is 5+8 = <<5+8=13>>13. + 

        Step 2: The waiter ate 13 x 8 / 13 = <<13*8/13=6>>6 slices of the pizza. - 

        Step 3: Buzz ate 78 - 6 = <<78-6=72>>72 slices of the pizza. - 

        Step 4: The waiter ate 20 less than the number of slices that Buzz ate which is 72 - 20 = 52. - 

        Step 5: The waiter ate 52 slices of the pizza. The answer is: 52 -
        """
        label_steps = re.split("Step \d+:", d["label"])
        label_steps = [s.strip() for s in label_steps[1:]]
        try:  # 过滤格式不对的
            for s in label_steps:
                assert s[-1] in ["+", "-"], label_steps
        except:
            continue

        step_labels = [1 if l[-1] == "+" else 0 for l in label_steps]        
        # labels = adjust_labels(step_labels, startLabel=args.clip_max)
        try:
            assert len(steps) == len(step_labels)
        except:
            continue
        # if len(set(step_labels)) > 1:
        queries.append(
            {
                "query": instruction_format(question),
                "answer": f" {prm_token}\n".join(steps) + f" {prm_token}",  # 在每个步骤的结尾加上特殊token[PRM]
                "labels": step_labels,  # + [outcome_label],
            }
        )
        ids = tokenizer.encode(queries[-1]["query"] + queries[-1]["answer"])
        if len(ids) > 512 and len(ids) <= 1024:
            longer_queries.append(queries.pop())
        elif len(ids) > 1024:
            longest_queries.append(queries.pop())
    
    if accelerator.is_local_main_process:
        print(f"Data Examples:\n{queries[0]}\n{queries[-1]}")
        print(f"Dataset Length:{len(queries)}")
        print(statistic)

    return (
        queries,
        longer_queries,
        longest_queries,
    )  # 全部数据，5024-1024的数据，超过1024的数据


class TrainDataset(Dataset):
    def __init__(self, dataset1, dataset2, dataset3):
        # 64, 24, 8
        # bs1, bs2, bs3 = 64, 24, 8
        bs1, bs2, bs3 = 32, 16, 8
        self.iteration_1 = math.ceil(len(dataset1) / bs1)
        self.iteration_2 = math.ceil(len(dataset2) / bs2)
        self.iteration_3 = math.ceil(len(dataset3) / bs3)
        self.dataset = []
        for i in range(self.iteration_1): 
            self.dataset.append(dataset1[i * bs1 : (i + 1) * bs1])
        for i in range(self.iteration_2):
            self.dataset.append(dataset2[i * bs2 : (i + 1) * bs2])
        for i in range(self.iteration_3):
            self.dataset.append(dataset3[i * bs3 : (i + 1) * bs3])

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return self.iteration_1 + self.iteration_2 + self.iteration_3


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="/nobackup/hf-dataset-download/Math-Shepherd",
    )
    parser.add_argument(
        "--model-path", type=str, default="/nobackup/hf-model/deepseek-math-7b-base"
    )
    parser.add_argument(
        "--save-path", type=str, default="/nobackup/prm_checkpoints/neg-zeta-16"
    )
    parser.add_argument("--zeta", type=int, default=4)
    parser.add_argument(
        "--loss-type",
        type=str,
        default="rank",
        choices=["rank", "orm", "mse", "bce", "focal", "sce",  "rank_step", "rank_error",  "pairwise_ranking", "ranking_loss_with_error_order", 
        "ranking_loss_with_info_gain", "ranking_loss_td", "ranking_loss_with_reward_based_gain" , "ranking_loss_with_hard_negatives", 
        "rank_contrastive", "ranking_loss_one_error_step", "rank_contrastive_loss2", "rank_contrastive_temperature", "rank_contrastive_exp_before",
        "rank_contrastive_temperature_neg_mean", "rank_contrastive_temperature_new", "rank_contrastive_temperature_neg_sum", "rank_contrastive_temperature_min",
        "rank_orm"],
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=2e-6,        
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,        
    )

    parser.add_argument(
        "--wandb-project", type=str, default="prm"
    )
    parser.add_argument(
        "--wandb-runname", type=str, default="random"
    )    

    parser.add_argument("--clip-min", type=float, default=-1)
    parser.add_argument("--clip-max", type=float, default=1)

    parser.add_argument("--temperature", type=float, default=1)

    

    args = parser.parse_args()
    print(args)

    seed_everything(0)
    accelerator = Accelerator()
    prm_token = "[PRM]"

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        use_flash_attention_2=True,
        trust_remote_code=True,
    )
    tokenizer.add_special_tokens({"additional_special_tokens": [prm_token]})
    prm_token_id = tokenizer.encode(prm_token, add_special_tokens=False)[-1]
    # dataset1, dataset2, dataset3 = generate_dataset(prm_token, tokenizer)
    dataset1, dataset2, dataset3 = generate_dataset_with_coarse_data(prm_token, tokenizer)
    

    dataset = TrainDataset(dataset1, dataset2, dataset3)

    model.resize_token_embeddings(len(tokenizer))
    reward_model = AutoModelForCausalLMWithValueHead(model)

    def data_collator(example, tokenizer=tokenizer):
        inputs = []
        special_ids = []
        step_labels = []
        orm_tokens, orm_labels = [], []
        has_neg = []
        template = "{query}\n{answer}"
        example = example[0]
        for d in example:
            input_ids = tokenizer.encode(
                template.format(query=d["query"], answer=d["answer"]),
                add_special_tokens=False,
            )
            # print("input_ids:\t", input_ids)
            inputs.append(torch.tensor(input_ids))

            cur_special_ids = []
            for ii, id in enumerate(input_ids):
                if id == prm_token_id:  # prm token位置
                    cur_special_ids.append(ii)
            assert len(cur_special_ids) == len(d["labels"])
            special_ids.append(torch.tensor(cur_special_ids))
            step_labels.append(torch.tensor(d["labels"]))
            orm_tokens.append(cur_special_ids[-1])
            has_neg.append(1 if 0 in d["labels"] else 0)  # 是否有不正确的步骤

        try:
            inputs = pad_sequence(inputs, padding_value=tokenizer.pad_token_id, batch_first=True)            
            attention_mask = inputs!=tokenizer.pad_token_id
            special_ids = pad_sequence(special_ids, padding_value=0, batch_first=True)
            # print("step_labels", step_labels)
            step_labels = pad_sequence(step_labels, padding_value=-100, batch_first=True)
        except:
            print("example", example)
            print("inputs:", inputs)   
                                            
        return {
            "input_ids": inputs.int(),
            "attention_mask": attention_mask.int(),
            "special_tokens": special_ids,
            "step_labels": step_labels,
            "orm_tokens": torch.tensor(orm_tokens),
            "has_neg": torch.tensor(has_neg),
        }

    deepspeed_config = json.load(open("accelerate_configs/deepspeed_3.json"))
    deepspeed_config["scheduler"]["params"] = {
        "warmup_min_lr": 0,
        "warmup_max_lr": "auto",
        "warmup_num_steps": "auto",
        "total_num_steps": "auto",
    }

    training_args = TrainingArguments(
        output_dir=args.save_path,
        overwrite_output_dir=True,
        optim="adamw_torch",
        # learning_rate = 2e-6,  # 2e-6 for 8 GPUs
        learning_rate = args.lr,  # 2e-6 for 8 GPUs
        lr_scheduler_type="cosine",
        # warmup_steps = 150,
        warmup_ratio=0.1,
        gradient_checkpointing=True,
        num_train_epochs=1,
        gradient_accumulation_steps=1,  # 4 for 8 GPUs
        # per_device_train_batch_size=1,
        per_device_train_batch_size=args.batch_size,
        save_strategy="epoch",
        report_to="wandb",
        logging_strategy="steps",
        logging_steps=10,
        remove_unused_columns=False,
        bf16=True,
        fp16_backend="auto",
        disable_tqdm=False,
        save_safetensors=False,
        # group_by_length = True,
        deepspeed=deepspeed_config,
        # sharded_ddp="zero_dp_2",
    )
    
    trainer = PRMTrainer(
        reward_model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        loss_type=args.loss_type,
        wandb_project=args.wandb_project,
        wandb_runname=args.wandb_runname,
        accelerator=accelerator
    )

    trainer.train()
    if accelerator.is_local_main_process:
        trainer.save_model("args.save_path")
        tokenizer.save_pretrained("args.save_path")
         
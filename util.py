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
import re
import math
import argparse
import wandb
from moe import MOELayer

 

def merge_steps(steps, step_labels, k):  
    if k > len(steps) - 1:
        return [], []
    assert len(steps) == len(step_labels), "steps should equal with step_labels"
    
    merged_steps = []
    merged_labels = []
    
    last_index = len(steps) - 1

    for i in range(0, len(steps) - 1, k):
        end_index = min(i + k, len(steps) - 1)
                
        merged_text = " ".join([steps[j].split(":", 1)[1].strip() for j in range(i, end_index)])
        merged_steps.append(f"Step {len(merged_steps) + 1}: {merged_text}")
                
        if all(step_labels[j] == 1 for j in range(i, end_index)):
            merged_labels.append(1)
        else:
            merged_labels.append(1) 
    
    merged_steps.append(
        f"Step {len(merged_steps) + 1}: {steps[last_index].split(':', 1)[1].strip()}"
    )
    merged_labels.append(0)

    return merged_steps, merged_labels

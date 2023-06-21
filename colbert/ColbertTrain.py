from datasets import DatasetDict, load_from_disk
import os
import pandas as pd
import torch
import torch.nn.functional as F
from ColbertTokenizer import *
from ColbertModel import *
import json
import pickle
from transformers import (
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)
from omegaconf import OmegaConf

def main(cfg):
    set_seed(cfg.train.seed)
    args = TrainingArguments(
        output_dir = 'results/colbert',
        evaluation_strategy = 'epoch',
        learning_rate = cfg.model.LR,
        num_train_epochs = cfg.model.epoch,
        

    )








if name = "__main__":
    cfg = OmegaConf.load('ColbertConfig.yaml')
    main(cfg)

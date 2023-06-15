import logging
import os
import sys

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

import torch
from torch.utils.data import DataLoader

from datasets import DatasetDict, load_from_disk
import evaluate
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
from utils.Dataloader import  *
from utils.Model import *
import yaml
logger = logging.getLogger(__name__)

def train(cfg):
    # # 모델을 초기화하기 전에 난수를 고정합니다.
    # set_seed(training_args.seed)
    
    pl.seed_everything(cfg["train"]["seed"], workers=True)
    
    datasets = load_from_disk(cfg["data"]["dataset_name"])
    print(datasets)
    
    #save_path, folder_name = cfg['save_path'], cfg['folder_name']
    model_config = AutoConfig.from_pretrained(cfg['model']['model_name'])

    model = newModel(cfg['model']['model_name'],
                  model_config,
                  cfg['model']['LR'], 
                  cfg['model']['loss_function'], 
                  cfg['model']['optim'], 
                  cfg['model']['scheduler'],
                  cfg
                  )
    
    # AutoConfig를 이용하여 tokenizer를 불러옵니다.
    tokenizer = AutoTokenizer.from_pretrained(
        cfg['model']['model_name'], max_length = 200
    )
    
    run_mrc(cfg, datasets, tokenizer, model)
        

def run_mrc(
    cfg,
    datasets: DatasetDict,
    tokenizer,
    model,
) -> None:
    print("MRC start")
    dataloader = MRCDataModule(cfg, datasets, tokenizer, model)
    print("MRC Done")

    print("Train Dataset:", dataloader.train_dataset)
    print("Eval Dataset:", dataloader.eval_dataset)
    
    trainer = pl.Trainer(accelerator='gpu', max_epochs=cfg["model"]["epoch"])
    
    trainer.fit(model = model, datamodule = dataloader)
    trainer.test(model=model, datamodule = dataloader)
    torch.save(model, f'model_1.pt')

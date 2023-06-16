import os
import sys

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

import torch
from pytorch_lightning.loggers import WandbLogger
import wandb
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor

from datasets import DatasetDict, load_from_disk
import evaluate
from transformers import (
    AutoConfig,
    AutoTokenizer,
)
from utils.Dataloader import MRCDataModule, Dataset
from utils.Model import newModel
import yaml

def train(cfg):
    # # 모델을 초기화하기 전에 난수를 고정합니다.
    # set_seed(training_args.seed)
    save_path, folder_name = cfg['save_path'], cfg['folder_name']
    
    pl.seed_everything(cfg["train"]["seed"], workers=True)
    
    datasets = load_from_disk(cfg["model"]["train_path"])
    print(datasets)
    
    #save_path, folder_name = cfg['save_path'], cfg['folder_name']
    model_config = AutoConfig.from_pretrained(pretrained_model_name_or_path = cfg['model']['model_name'])

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
    
    run_mrc(cfg, datasets, tokenizer, model, save_path, folder_name)
        

def run_mrc(
    cfg,
    datasets: DatasetDict,
    tokenizer,
    model,
    save_path,
    folder_name
) -> None:
    print("MRC start")
    dataloader = MRCDataModule(cfg, datasets, tokenizer, model)
    print("MRC Done")

    print("Train Dataset:", dataloader.train_dataset)
    print("Eval Dataset:", dataloader.eval_dataset)
    
    wandb.init(name = folder_name, project = "MRC", entity = "hypesalmon", dir = save_path)
    wandb_logger = WandbLogger(save_dir = save_path)
    wandb_logger.experiment.config.update(cfg)
    
    early_stopping = EarlyStopping(
        monitor = cfg['EarlyStopping']['monitor'],
        min_delta = cfg['EarlyStopping']['min_delta'],
        patience=cfg['EarlyStopping']['patience'],
        verbose=cfg['EarlyStopping']['verbose'],
        mode='max',
    )
    
    checkpoint = ModelCheckpoint(
        dirpath ='./checkpoints/',
        filename = cfg['model']['model_name']+'-{epoch}-{val_em:.2f}-{val_f1:.2f}',
        every_n_epochs = 1
    )

    # learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    trainer = pl.Trainer(accelerator='gpu',
                         max_epochs=cfg["model"]["epoch"],
                         log_every_n_steps = 1,
                         logger = wandb_logger,
                         callbacks = [early_stopping, checkpoint, lr_monitor] if cfg['EarlyStopping']['turn_on'] else [checkpoint])
    
    trainer.fit(model = model, datamodule = dataloader)
    trainer.test(model=model, datamodule = dataloader)
    torch.save(model, f'{save_path}/{folder_name}_model.pt')

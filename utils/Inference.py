import logging
import os
import sys

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

import torch
from itertools import chain
from typing import Callable, Dict, List, Tuple
from datasets import DatasetDict, load_from_disk, Features, Sequence, Value
import evaluate
from utils.Dataloader import MRCDataModule
from utils.Model import newModel
from utils.utils_qa import post_processing_function
import yaml
logger = logging.getLogger(__name__)
from transformers import AutoTokenizer

def inference(cfg):
    save_path, folder_name = cfg['save_path'], cfg['folder_name']
    datasets = load_from_disk(cfg["model"]["test_path"])
    print(datasets)

    model = torch.load(f'{save_path}/{folder_name}_model.pt')
    #model = torch.load(f'./results/2023-06-16-21:35:48_HYPE연어/2023-06-16-21:35:48_HYPE연어_model.pt')
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
    
    predicts = trainer.predict(model = model, datamodule = dataloader)
    
    start_logits = torch.cat([x["start_logits"] for x in predicts])
    end_logits = torch.cat([x["end_logits"] for x in predicts])
    predictions = (start_logits, end_logits)
    
    ids = [x["id"] for x in predicts]
    id = list(chain(*ids))
    
    preds = post_processing_function("predict", cfg, id, predictions, tokenizer)


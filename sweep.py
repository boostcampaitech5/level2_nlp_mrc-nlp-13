import yaml
import os
import pickle as pickle
import torch
import pytorch_lightning as pl
import wandb
from transformers import AutoConfig, AutoTokenizer
from datasets import DatasetDict, load_from_disk

from utils.utils_qa import get_folder_name
from utils.Dataloader import MRCDataModule, Dataset
from utils.Model import newModel
from utils.Inference import *

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor


if __name__ == '__main__':
    with open('config.yaml') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
        
    get_folder_name(cfg)
    save_path, folder_name = cfg['save_path'], cfg['folder_name']

    pl.seed_everything(cfg["train"]["seed"], workers = True)
    
    # config의 sweep부분을 불러옵니다.
    sweep_config = cfg['sweep']
    ver = 0

    # set version to save model

    def train():
        '''
        기존의 train 함수와 다른 것은 없으나, sweep으로 조정하고 싶은 변수는
        cfg['model']['parameter'] 형식에서 config.parameter 로 바꾸어 입력해주세요!
        '''
        global ver
        global save_path
        cfg['version'] = ver
            # logger 생성
        '''
        pip install wandb
        첫 실행할 때 로그인을 위해 본인 api key를 복사해서 붙여넣어주세요
        wandblog는 results 폴더에 실행모델과 함께 저장됩니다
        '''
        wandb.init(name=folder_name, project="MRC", entity="Hype연어", dir=save_path)
        wandb_logger = WandbLogger(save_dir=save_path)
        wandb_logger.experiment.config.update(cfg)

        config = wandb.config
        
        datasets = load_from_disk(cfg["model"]["train_path"])
        
        '''모델 설정은 기본 설정을 그대로 가져오고 사용하는 레이블의 개수만 현재 데이터에 맞춰서 설정'''
        model_config = AutoConfig.from_pretrained(cfg['model']['model_name'])
        model = newModel(cfg['model']['model_name'],
                    model_config,
                    cfg["model"]["LR"], 
                    cfg['model']['loss_function'], 
                    cfg['model']['optim'], 
                    cfg['model']['scheduler'],
                    cfg)

        tokenizer = AutoTokenizer.from_pretrained(
            cfg["model"]["model_name"], max_length = 200
        )
        
        dataloader = MRCDataModule(cfg, datasets, tokenizer, model)
        
        early_stopping = EarlyStopping(
            monitor = cfg['EarlyStopping']['monitor'],
            min_delta=cfg['EarlyStopping']['min_delta'],
            patience=cfg['EarlyStopping']['patience'],
            verbose=cfg['EarlyStopping']['verbose'],
            mode='max',
        )

        checkpoint = ModelCheckpoint(
            dirpath ='./checkpoints/',
            filename = cfg['model']['model_name']+'-{epoch}-{val_em:.2f}-{val_f1:.2f}',
            every_n_epochs = 1)

        lr_monitor = LearningRateMonitor(logging_interval='step')

        trainer = pl.Trainer(accelerator = "auto",
                            max_epochs = cfg['model']['epoch'],
                            log_every_n_steps = 1,
                            logger = wandb_logger,
                            callbacks=[early_stopping, checkpoint,lr_monitor] if cfg['EarlyStopping']['turn_on'] else [checkpoint],
                            precision='16-mixed') #fp16 사용
        
        trainer.fit(model=model, datamodule=dataloader)
        trainer.test(model=model, datamodule=dataloader)
        
        torch.save(model, f'{save_path}/{folder_name}_ver{ver}_model.pt')
        ver += 1


    sweep_id = wandb.sweep(sweep = sweep_config, project = 'MRC_Sweeps')
    wandb.agent(sweep_id=sweep_id, function=train, count=cfg['sweepcnt'])
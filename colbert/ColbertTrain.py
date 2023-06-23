import torch
from colbert.ColbertTokenizer import *
from colbert.ColbertModel import *
from transformers import (
    AutoConfig,
    set_seed,
)
from omegaconf import OmegaConf

def ColbertTrain(cfg):
    cfg = OmegaConf.create(cfg)
    save_path, folder_name = cfg.save_path, cfg.folder_name
    set_seed(cfg.train.seed)
    model_config = AutoConfig.from_pretrained('klue/bert-base')    

    Model = ColbertModel(model_config, cfg.model.LR, cfg.model.optim, cfg.data.loss, cfg.model.batch_size, cfg.data.top_k_retrieval)
    dataloader = ColbertDataModule(cfg)

    trainer = pl.Trainer(accelerator='gpu',
                         max_epochs=cfg.model.epoch,
                         log_every_n_steps = 1,
                         precision = '16-mixed')

    trainer.fit(model=Model, datamodule = dataloader)
    torch.save(Model, f'{save_path}/{folder_name}_model.pt')



if __name__ == "__main__":
    cfg = OmegaConf.load('ColbertConfig.yaml')
    #train(cfg)

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
from torch.utils.data import DataLoader
from utils.Utils import *
from datasets import load_from_disk

class Dataset(torch.utils.data.Dataset):
    """ Dataset 구성을 위한 class."""
    def __init__(self, dataset, stage):#inputs, attention_masks, token_type_ids,start_positions,end_positions,
        self.dataset = dataset
        self.stage = stage
        
    def __getitem__(self, idx):
        if self.stage == 'fit':
            item = {key: torch.tensor(val) for key, val in self.dataset[idx].items()}
            return item
        else:
            item = {
                "input_ids": torch.tensor(self.dataset[idx]["input_ids"]),
                "attention_mask": torch.tensor(self.dataset[idx]["attention_mask"]),
                "offset_mapping": torch.tensor(self.dataset[idx]["offset_mapping"]),
            }
            id = self.dataset[idx]["example_id"]

            return item, id
        
    def __len__(self):
        return len(self.dataset)

class MRCDataModule(pl.LightningDataModule):
    def __init__(self, cfg, datasets, tokenizer, model):
        super().__init__()
        self.config = cfg
        self.datasets = datasets
        self.tokenizer = tokenizer
        self.model = model
        self.batch_size = cfg["model"]["batch_size"]
        self.train_dataset=None
        self.eval_dataset=None
        self.test_dataset = None
        self.column_names=None
        self.question_column_name=None
        self.answer_column_name=None
        self.context_column_name=None
        self.pad_on_right=None
        self.last_checkpoint=None
        self.max_seq_length=cfg["data"]["max_seq_length"]

        self.shuffle = True
        self.num_workers = 8

    def setup(self,stage='fit'):
        print("prepare start")
        # Padding에 대한 옵션을 설정합니다.
        # (question|context) 혹은 (context|question)로 세팅 가능합니다.
        self.pad_on_right = self.tokenizer.padding_side == "right"

        # 오류가 있는지 확인합니다.
        # self.last_checkpoint, self.max_seq_length = check_no_error(
        #     self.data_args, self.training_args, self.datasets, self.tokenizer
        # )
        print("prepare done")

        print("setup Start")

        if stage == 'fit':
            self.train_dataset = self.datasets["train"]
            self.column_names = self.datasets["train"].column_names
            self.question_column_name = "question" if "question" in self.column_names else self.column_names[0]
            self.context_column_name = "context" if "context" in self.column_names else self.column_names[1]
            self.answer_column_name = "answers" if "answers" in self.column_names else self.column_names[2]
            
            # dataset에서 train feature를 생성합니다.
            self.train_dataset = self.train_dataset.map(
                prepare_train_features,
                batched=True,
                num_proc=self.config["data"]["preprocessing_num_workers"],
                remove_columns=self.column_names,
                load_from_cache_file=not self.config["data"]["overwrite_cache"],
                fn_kwargs = {"tokenizer":self.tokenizer, "config":self.config}
            )
            print(self.train_dataset)
            self.train_dataset = Dataset(self.train_dataset, stage)

            self.eval_dataset = self.datasets["validation"]
            self.column_names = self.datasets["validation"].column_names
            self.question_column_name = "question" if "question" in self.column_names else self.column_names[0]
            self.context_column_name = "context" if "context" in self.column_names else self.column_names[1]
            self.answer_column_name = "answers" if "answers" in self.column_names else self.column_names[2]
            
            # Validation Feature 생성
            self.eval_dataset = self.eval_dataset.map(
                prepare_validation_features,
                batched=True,
                num_proc=self.config["data"]["preprocessing_num_workers"],
                remove_columns=self.column_names,
                load_from_cache_file=not self.config["data"]["overwrite_cache"],
                fn_kwargs = {"tokenizer":self.tokenizer, "config":self.config}
            )
            self.eval_dataset = Dataset(self.eval_dataset,  stage= "eval")
            print("setup Done")
            
        if stage == 'test':
            self.test_dataset = self.datasets["validation"]
            self.column_names = self.datasets["validation"].column_names
            self.question_column_name = "question" if "question" in self.column_names else self.column_names[0]
            self.context_column_name = "context" if "context" in self.column_names else self.column_names[1]
            self.answer_column_name = "answers" if "answers" in self.column_names else self.column_names[2]

            # Validation Feature 생성
            self.test_dataset = self.test_dataset.map(
                prepare_validation_features,
                batched=True,
                num_proc=self.config["data"]["preprocessing_num_workers"],
                remove_columns=self.column_names,
                load_from_cache_file=not self.config["data"]["overwrite_cache"],
                fn_kwargs = {"tokenizer":self.tokenizer, "config":self.config}
            )
            print(self.test_dataset)
            self.test_dataset = Dataset(self.test_dataset, stage)
            
    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=self.shuffle, num_workers = self.num_workers)
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.eval_dataset, batch_size=self.batch_size, num_workers = self.num_workers)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size = self.batch_size, num_workers = self.num_workers)
    
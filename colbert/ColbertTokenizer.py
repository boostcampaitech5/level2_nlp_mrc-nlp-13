from transformers import AutoTokenizer
from datasets import DatasetDict, load_from_disk
from torch.utils.data import TensorDataset
from omegaconf import OmegaConf
import pytorch_lightning as pl
import torch
import json

class ColbertDataset(torch.utils.data.Dataset):
    def __init__(self, query, doc, stage):
        self.query = query
        self.doc = doc
        self.stage = stage

    def __getitem__(self, idx):
            query = {
                "input_ids": torch.tensor(self.query[idx][0]),
                "attention_mask": torch.tensor(self.query[idx][1]),
                "token_type_ids": torch.tensor(self.query[idx][2]),
            }
            doc = {
                "input_ids": torch.tensor(self.doc[idx][0]),
                "attention_mask": torch.tensor(self.doc[idx][1]),
                "token_type_ids": torch.tensor(self.doc[idx][2]),
            }
            return query, doc
        
    def __len__(self):
        return len(self.query)



class ColbertDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = 'klue/bert-base'
        self.train_dir = config.model.train_path
        self.test_dir = config.model.test_path
        self.tokenizer = None

        self.batch_size = config.model.batch_size


        self.train_dataset=None
        self.eval_dataset=None
        self.test_dataset = None
        self.predict_dataset = None
        self.column_names=None
        self.question_column_name=None
        self.answer_column_name=None
        self.context_column_name=None
        self.pad_on_right=None
        self.last_checkpoint=None
        self.max_seq_length=config.data.max_seq_length

        self.shuffle = True
        self.num_workers = 0

    def prepare_tokens(self, dataset, stage=False):
        preprocessed_query=[]
        for query in dataset['question']:
            preprocessed_query.append('[Q] '+query)

        q = self.tokenizer(
            preprocessed_query,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
            return_token_type_ids=True,
            )
        
        preprocessed_doc=[]
        for doc in dataset['context']:
            preprocessed_doc.append('[D] '+doc)
        d = self.tokenizer(
            preprocessed_doc,
            return_tensors="pt",
            padding='max_length',
            truncation=True,
            return_token_type_ids=True,
            )
    
        return q, d

    def setup(self,stage='fit'):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model)
        self.tokenizer.add_special_tokens({'additional_special_tokens': ['[Q]','[D]']})

        if stage == 'fit':
            self.train_dataset = load_from_disk(self.train_dir)['train']
            self.column_names = self.train_dataset.column_names
            q, d = self.prepare_tokens(self.train_dataset)
            q = TensorDataset(q['input_ids'], q['attention_mask'], q['token_type_ids'])
            d = TensorDataset(d['input_ids'], d['attention_mask'], d['token_type_ids'])
            self.train_dataset = ColbertDataset(q, d, stage)


            self.eval_dataset = load_from_disk(self.train_dir)["validation"]
            self.column_names = self.eval_dataset.column_names
            q, d = self.prepare_tokens(self.eval_dataset)
            q = TensorDataset(q['input_ids'], q['attention_mask'], q['token_type_ids'])
            d = TensorDataset(d['input_ids'], d['attention_mask'], d['token_type_ids'])
            self.eval_dataset = ColbertDataset(q, d,  stage = "eval")

        if stage == 'test':
            self.test_dataset = load_from_disk(self.train_dir)["validation"]
            self.column_names = self.test_dataset.column_names
            q, d = self.prepare_tokens(self.test_dataset)
            q = TensorDataset(q['input_ids'], q['attention_mask'], q['token_type_ids'])
            d = TensorDataset(d['input_ids'], d['attention_mask'], d['token_type_ids'])
            self.test_dataset = ColbertDataset(q, d, stage)

        if stage == 'predict':
            predict_query = load_from_disk(self.test_dir)['validation']
            self.column_names = predict_query.column_names
            with open('data/wikipedia_documents.json', "r", encoding="utf-8") as f:
                wiki = json.load(f)
            predict_doc = list(dict.fromkeys([v["text"] for v in wiki.values()]))
            self.predict_dataset = {"question" : predict_query['question'], "context" : predict_doc}
            q, d = self.prepare_tokens(self.predict_dataset)
            q = TensorDataset(q['input_ids'], q['attention_mask'], q['token_type_ids'])
            d = TensorDataset(d['input_ids'], d['attention_mask'], d['token_type_ids'])
            self.predict_dataset = ColbertDataset(q, d, stage)


    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=self.shuffle, num_workers = self.num_workers)
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.eval_dataset, batch_size=self.batch_size, num_workers = self.num_workers)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers = self.num_workers)
    
    def predict_dataloader(self):
        return torch.utils.data.DataLoader(self.predict_dataset, batch_size = self.batch_size, num_workers = self.num_workers)
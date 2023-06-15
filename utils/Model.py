import pytorch_lightning as pl
import torch
import evaluate
from itertools import chain
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, EvalPrediction, AutoConfig
from torch.optim.lr_scheduler import StepLR

from datasets import load_from_disk
from utils.Utils import *

def compute_metrics(p: EvalPrediction):
    metric = evaluate.load("squad")
    return metric.compute(predictions=p.predictions, references=p.label_ids)

class newModel(pl.LightningModule):
    def __init__(self, MODEL_NAME, model_config, lr, loss, optim, scheduler, config):
        """
        모델 생성

        Args:
            MODEL_NAME: 사용할 모델 이름
            model_config: 사용할 모델 config
            lr: 모델 learning rate
            loss: 모델 loss function
            optim: 모델 optimizer
            scheduler: 모델 learning rate scheduler
            train_path: train data 경로
            test_path: test data 경로
        """
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.validation_step_outputs = []
        self.test_step_outputs = []
        
        self.MODEL_NAME = MODEL_NAME
        self.model_config = model_config
        self.lr = lr
        self.optim = optim
        self.scheduler = scheduler
        
        self.plm = AutoModelForQuestionAnswering.from_pretrained(pretrained_model_name_or_path = self.MODEL_NAME)
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME, max_length = 200)
        
        self.loss_dict = {
            'CrossEntropyLoss' : torch.nn.CrossEntropyLoss()
        }
        self.loss_func = self.loss_dict[loss]
        
    def forward(self, x):
        """
        model gets x -> predict start logits and end logits
        """
        x = self.plm(input_ids = x["input_ids"],
                        attention_mask = x["attention_mask"])
        return x["start_logits"], x["end_logits"]
        
        # x = self.plm(input_ids=x[0], attention_mask=x[2], token_type_ids=x[1])
        # return x["start_logits"], x["end_logits"]
    
    def training_step(self, batch):
        """
        calculate train loss
        """
        start_logits, end_logits = self(batch)
        start_positions, end_positions = batch["start_positions"], batch["end_positions"]
        
        start_loss = self.loss_func(start_logits, start_positions)
        end_loss = self.loss_func(end_logits, end_positions)
        loss = (start_loss + end_loss) / 2
        self.log("train_loss", loss)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """
        predict start logits and end logits
        """
        data, id = batch
        start_logits, end_logits = self(data)
        
        self.validation_step_outputs.append({"start_logits" : start_logits, "end_logits" : end_logits, "id" : id})
        
    def on_validation_epoch_end(self):
        """
        calculate validation score
        """
        outputs = self.validation_step_outputs
        start_logits = torch.cat([x["start_logits"] for x in outputs])
        end_logits = torch.cat([x["end_logits"] for x in outputs])
        predictions = (start_logits, end_logits)
        
        ids = [x["id"] for x in outputs]
        id = list(chain(*ids))
        
        preds = post_processing_function("eval", self.config, id, predictions, self.tokenizer)
        
        result = compute_metrics(preds)
        self.log("val_em", result["exact_match"])
        self.log("val_f1", result["f1"])
        self.validation_step_outputs.clear()
        
    def test_step(self, batch, batch_idx):
        """
        predict start logits and end logits
        """
        data, id = batch
        start_logits, end_logits = self(data)
        
        self.test_step_outputs.append({"start_logits" : start_logits, "end_logits" : end_logits, "id" : id})
    
    def on_test_epoch_end(self):
        """
        calculate test data score
        """
        outputs = self.test_step_outputs
        start_logits = torch.cat([x["start_logits"] for x in outputs])
        end_logits = torch.cat([x["end_logits"] for x in outputs])
        predictions = (start_logits, end_logits)
        
        ids = [x["id"] for x in outputs]
        id = list(chain(*ids))
        
        preds = post_processing_function("eval", self.config, id, predictions, self.tokenizer)
        
        result = compute_metrics(preds)
        self.log("test_em", result["exact_match"])
        self.log("test_f1", result["f1"])
        self.test_step_outputs.clear()
        
    def predict_step(self, batch, batch_idx):
        """
        model gets test data->predict start logits and end logits
        """
        data, id = batch
        start_logits, end_logits = self(data)
        
        return {"start_logits" : start_logits, "end_logits" : end_logits, "id" : id}
    
    def configure_optimizers(self):
        """
        select optimizer and learning rate scheduler
        """
        self.optimizer_dict = {
            'AdamW' : torch.optim.AdamW(self.parameters(), lr = self.lr)
        }
        optimizer = self.optimizer_dict[self.optim]
        self.lr_scheduler_dict = {
            'StepLR' : StepLR(optimizer, step_size = 1, gamma = 0.5)
        }
        
        if self.scheduler == 'None':
            return optimizer
        else:
            scheduler = self.lr_scheduler_dict[self.scheduler]
            return [optimizer], [scheduler]
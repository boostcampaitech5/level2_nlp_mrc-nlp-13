from tqdm.auto import tqdm
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from sklearn.metrics import top_k_accuracy_score
from transformers import (BertModel,
                          BertPreTrainedModel,
                          AutoTokenizer
                        )       
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class ColbertModel(pl.LightningModule):
    def __init__(self, config, lr, optim, loss, batch_size, topk):
        super().__init__()
        self.topk = topk
        self.lr = lr
        self.optim = optim
        self.loss_dict = {
            'CrossEntropyLoss' : torch.nn.CrossEntropyLoss(),
            'NllLoss' : F.nll_loss
        }
        self.loss_func = self.loss_dict[loss]
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained('klue/bert-base')
        self.tokenizer.add_special_tokens({'additional_special_tokens': ['[Q]','[D]']})
        
        self.dim = 128
        self.bert = BertModel(config)
        self.bert.resize_token_embeddings(len(self.tokenizer))
        self.linear = torch.nn.Linear(config.hidden_size, self.dim, bias=False)
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            module.weight.data.normal_(mean=0.0, std=1.0)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, x):
        Q, D = x
        Q = self.query(Q['input_ids'], Q['attention_mask'], Q['token_type_ids'])
        D = self.doc(D['input_ids'], D['attention_mask'], D['token_type_ids'])
        scores = self.get_scores(Q, D)
        return scores
    
    def query(self, input_ids, attention_mask, token_type_ids):
        Q = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)[0]
        #print(Q)
        Q = self.linear(Q)
        Q = F.normalize(Q, p=2, dim=2)
        return Q
    
    def doc(self, input_ids, attention_mask, token_type_ids):
        D = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)[0]
        D = self.linear(D)
        D = F.normalize(D, p=2, dim=2)

        return D

    def get_scores(self, Q, D, eval=False):
        if eval:
                final_score=torch.tensor([])
                for D_batch in tqdm(D):
                    D_batch = torch.Tensor(D_batch)
                    p_seqeunce_output=D_batch.transpose(1,2) #(batch_size,hidden_size,p_sequence_length)
                    q_sequence_output=Q.view(16,1,-1,self.dim) #(batch_size, 1, q_sequence_length, hidden_size)
                    dot_prod = torch.matmul(q_sequence_output,p_seqeunce_output) #(batch_size,batch_size, q_sequence_length, p_seqence_length)
                    max_dot_prod_score =torch.max(dot_prod, dim=3)[0] #(batch_size,batch_size,q_sequnce_length)
                    score = torch.sum(max_dot_prod_score,dim=2)#(batch_size,batch_size)
                    final_score = torch.cat([final_score,score],dim=1)
                #print(final_score.size())
                return final_score

        else:
                p_seqeunce_output=D.transpose(1,2) #(batch_size,hidden_size,p_sequence_length)
                q_sequence_output=Q.view(self.batch_size,1,-1,self.dim) #(batch_size, 1, q_sequence_length, hidden_size)
                dot_prod = torch.matmul(q_sequence_output,p_seqeunce_output) #(batch_size,batch_size, q_sequence_length, p_seqence_length)
                max_dot_prod_score =torch.max(dot_prod, dim=3)[0] #(batch_size,batch_size,q_sequnce_length)
                final_score = torch.sum(max_dot_prod_score,dim=2)#(batch_size,batch_size)
            
                return final_score
    
    def training_step(self, batch):
        scores = self(batch)
        sim_scores = F.log_softmax(scores, dim=1).cuda()

        targets = torch.arange(0, self.batch_size).long().cuda()
        loss = self.loss_func(sim_scores, targets)
        self.log("train loss", loss)
        return loss
        
    def predict_step(self, Q, D):
        """
        model gets test data->predict start logits and end logits
        """
        Q, D = Q[0], Q[1]
        Q = self.query(Q['input_ids'], Q['attention_mask'], Q['token_type_ids']).to('cpu')
        batched_p_embs = []
        p_embs=[]
        for step, idx in enumerate(tqdm(range(len(D)))):
            p_emb = self.doc(D['input_ids'][idx].unsqueeze(0), D['attention_mask'][idx].unsqueeze(0), D['token_type_ids'][idx].unsqueeze(0)).to('cpu').numpy()
            p_embs.append(p_emb)
            if (step+1)%200 ==0:
                batched_p_embs.append(p_embs)
                p_embs=[]
        batched_p_embs.append(p_embs)
        scores = self.get_scores(Q, p_embs, eval=True)
        sim_scores = F.log_softmax(scores, dim=1).cuda()
        
        return sim_scores

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
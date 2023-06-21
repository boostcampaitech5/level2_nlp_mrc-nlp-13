from tqdm.auto import tqdm
import torch
import torch.nn.functional as F
from transformers import (BertModel,
                          BertPretrainedModel,
                        )       

class ColbertModel(BertPretrainedModel):
    def __init__(self, config):
        super(ColbertModel, self).__init__(config)
        
        self.dim = 128
        self.bert = BertModel(config)
        self.linear = torch.nn.linear(config.hidden_size, self.dim, bias=False)
        self.init_weights()

    def forward(self, Q, D):
        Q = self.query(*Q)
        D = self.doc(*D)
        scores = self.get_scores(Q, D)
        return scores
    
    def query(self, input_ids, attention_mask, token_type_ids):
        Q = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)[0]
        Q = self.linear(Q)
        Q = F.normalize(Q, p=2, dim=2)
        return Q
    
    def doc(self, input_ids, attention_mask, token_type_ids):
        D = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)[0]
        D = self.linear(D)
        D = F.normalize(D, p=2, dim=2)

        return D

    def get_scores(self, Q, D):
        final_score = torch.tensor([])
        for batch in tqdm(D):
            batch = torch.Tensor(batch).squeeze()
            p_output = batch.transpose(1,2)
            q_output = Q.view(240, 1, -1, self.dim)
            dot_prod = torch.matmul(p_output, q_output)
            max_dot_prod = torch.max(dot_prod, dim=3)[0]
            score = torch.sum(max_dot_prod, dim=2)
            final_score = torch.cat([final_score, score], dim=1)
                                    
        return final_score
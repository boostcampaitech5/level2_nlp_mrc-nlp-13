import torch
from colbert.ColbertTokenizer import *
from colbert.ColbertModel import *
from transformers import (
    AutoConfig,
    set_seed,
)
from omegaconf import OmegaConf
import os
import json
import pandas as pd

def ColbertInference(cfg, dataset, data_path, context_path):
    cfg = OmegaConf.create(cfg)
    # save_path, folder_name = cfg.save_path, cfg.folder_name
    set_seed(cfg.train.seed)
    Model = torch.load(f'results/2023-06-22-16:18:33_HYPE연어/2023-06-22-16:18:33_HYPE연어_model.pt')
    k = cfg.data.top_k_retrieval

    # with open('/opt/ml/input/data/wikipedia_documents.json', "r", encoding="utf-8") as f:
    #     wiki = json.load(f)
    # context = list(dict.fromkeys([v["text"] for v in wiki.values()]))

    # test_dataset = load_from_disk(model.cfg.test_path)['validation']
    # query= list(test_dataset['question'])
    # mrc_ids =test_dataset['id']
    # length = len(test_dataset)


    dataloader = ColbertDataModule(cfg)

    trainer = pl.Trainer(accelerator='gpu')
    predicts = trainer.predict(model = Model, datamodule = dataloader)
    print(predicts)
    rank = torch.argsort(predicts).squeeze()
    doc_scores, doc_indices = [], []
    doc_scores.append(predicts[rank].tolist()[:k])
    doc_indices.append(rank.tolist()[:k])

    with open(os.path.join(data_path, context_path), "r", encoding="utf-8") as f:
            wiki = json.load(f)
            
    contexts = list(
            dict.fromkeys([v["text"] for v in wiki.values()])
            )  # set 은 매번 순서가 바뀌므로

    total = []
    for idx, example in enumerate(
                tqdm(dataset, desc="Colbert retrieval: ")
            ):
                tmp = {
                    # Query와 해당 id를 반환합니다.
                    "question": example["question"],
                    "id": example["id"],
                    # Retrieve한 Passage의 id, context를 반환합니다.
                    "context": " ".join(
                        [contexts[pid] for pid in doc_indices[idx]]
                    ),
                }
                if "context" in example.keys() and "answers" in example.keys():
                    # validation 데이터를 사용하면 ground_truth context와 answer도 반환합니다.
                    tmp["original_context"] = example["context"]
                    tmp["answers"] = example["answers"]
                total.append(tmp)

    cqas = pd.DataFrame(total)
    return cqas
    
if __name__ == "__main__":
    cfg = OmegaConf.load('ColbertConfig.yaml')
    main(cfg)

import os

import json
import tqdm
from tqdm.auto import tqdm
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import numpy as np
import collections
import wandb
from pytorch_lightning.loggers import WandbLogger
import torch
from itertools import chain
from typing import Callable, Dict, List, Tuple, Optional
from datasets import DatasetDict, load_from_disk, Features, Sequence, Value, Dataset, concatenate_datasets
import evaluate
from utils.Retrieval import SparseRetrieval
from utils.bm25 import BM25Retrieval
import utils.Dataloader
from utils.Model import compute_metrics
from utils.utils_qa import get_folder_name, prepare_predict_features
import yaml
from transformers import AutoTokenizer, AutoConfig, AutoModelForQuestionAnswering, EvalPrediction
from torch.optim.lr_scheduler import StepLR

class Dataloader(pl.LightningDataModule):
    def __init__(self, cfg, datasets, tokenizer, model):
        super().__init__()
        self.config = cfg
        self.tokenizer = tokenizer
        self.model = model
        self.test_dataset = datasets
        self.batch_size = cfg["model"]["batch_size"]
        self.column_names=None
        self.question_column_name=None
        self.answer_column_name=None
        self.context_column_name=None
        self.pad_on_right=None
        self.last_checkpoint=None
        self.max_seq_length=cfg["data"]["max_seq_length"]
        self.num_workers = 8
        
    def setup(self, stage = 'fit'):
        """
        train data 기준으로 test해서 f1 score를 뽑아내기 위한 Dataloader
        """
        if stage == 'test':
            self.test_dataset = self.test_dataset["validation"]
            self.column_names = self.test_dataset.column_names

            self.question_column_name = "question" if "question" in self.column_names else self.column_names[0]
            self.context_column_name = "context" if "context" in self.column_names else self.column_names[1]
            self.answer_column_name = "answers" if "answers" in self.column_names else self.column_names[2]

            # Validation Feature 생성
            self.test_dataset = self.test_dataset.map(
                prepare_predict_features,
                batched=True,
                num_proc=self.config["data"]["preprocessing_num_workers"],
                remove_columns=self.column_names,
                load_from_cache_file=not self.config["data"]["overwrite_cache"],
                fn_kwargs = {"tokenizer":self.tokenizer, "config":self.config}
            )
            print(self.test_dataset)
            self.test_dataset = utils.Dataloader.Dataset(self.test_dataset, stage)
            
    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size = self.batch_size, num_workers = self.num_workers)

class Model(pl.LightningModule):
    def __init__(self, MODEL_NAME, model_config, lr, loss, optim, scheduler, config):
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
            'CrossEntropyLoss' : torch.nn.CrossEntropyLoss(),
        }
        self.loss_func = self.loss_dict[loss]
        
    def forward(self, x):
        """
        model gets x -> predict start logits and end logits
        """
        x = self.plm(input_ids = x["input_ids"],
                        attention_mask = x["attention_mask"])
        return x["start_logits"], x["end_logits"]
    
    def test_step(self, batch, batch_idx):
        """
        predict start logits and end logits
        """
        data, id = batch
        start_logits, end_logits = self(data)
        
        self.test_step_outputs.append({"start_logits" : start_logits, "end_logits" : end_logits, "id" : id})
    
    def on_test_epoch_end(self):
        """
        calculate test data score and record f1 score
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

        f1.append(result["f1"])
        self.test_step_outputs.clear()

def postprocess_qa_predictions(
    mode,
    examples,
    features,
    id,
    predictions: Tuple[np.ndarray, np.ndarray],
    version_2_with_negative: bool = False,
    n_best_size: int = 20,
    max_answer_length: int = 30,
    null_score_diff_threshold: float = 0.0,
    prefix: Optional[str] = None,
    is_world_process_zero: bool = True,
    save_path = ''
):
    """
    Post-processes : qa model의 prediction 값을 후처리하는 함수
    모델은 start logit과 end logit을 반환하기 때문에, 이를 기반으로 original text로 변경하는 후처리가 필요함

    Args:
        examples: 전처리 되지 않은 데이터셋 (see the main script for more information).
        features: 전처리가 진행된 데이터셋 (see the main script for more information).
        predictions (:obj:`Tuple[np.ndarray, np.ndarray]`):
            모델의 예측값 :start logits과 the end logits을 나타내는 two arrays              첫번째 차원은 :obj:`features`의 element와 갯수가 맞아야함.
        version_2_with_negative (:obj:`bool`, `optional`, defaults to :obj:`False`):
            정답이 없는 데이터셋이 포함되어있는지 여부를 나타냄
        n_best_size (:obj:`int`, `optional`, defaults to 20):
            답변을 찾을 때 생성할 n-best prediction 총 개수
        max_answer_length (:obj:`int`, `optional`, defaults to 30):
            생성할 수 있는 답변의 최대 길이
        null_score_diff_threshold (:obj:`float`, `optional`, defaults to 0):
            null 답변을 선택하는 데 사용되는 threshold
            : if the best answer has a score that is less than the score of
            the null answer minus this threshold, the null answer is selected for this example (note that the score of
            the null answer for an example giving several features is the minimum of the scores for the null answer on
            each feature: all features must be aligned on the fact they `want` to predict a null answer).
            Only useful when :obj:`version_2_with_negative` is :obj:`True`.
        output_dir (:obj:`str`, `optional`):
            아래의 값이 저장되는 경로
            dictionary : predictions, n_best predictions (with their scores and logits) if:obj:`version_2_with_negative=True`,
            dictionary : the scores differences between best and null answers
        prefix (:obj:`str`, `optional`):
            dictionary에 `prefix`가 포함되어 저장됨
        is_world_process_zero (:obj:`bool`, `optional`, defaults to :obj:`True`):
            이 프로세스가 main process인지 여부(logging/save를 수행해야 하는지 여부를 결정하는 데 사용됨)
    """
    assert (
        len(predictions) == 2
    ), "`predictions` should be a tuple with two elements (start_logits, end_logits)."
    all_start_logits, all_end_logits = predictions

    # example과 mapping되는 feature 생성
    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(id):
        features_per_example[example_id_to_index[feature]].append(i)

    # prediction, nbest에 해당하는 OrderedDict 생성합니다.
    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    if version_2_with_negative:
        scores_diff_json = collections.OrderedDict()

    # 전체 example들에 대한 main Loop
    for example_index, example in enumerate(tqdm(examples, desc='Postprocessing')):
        # 해당하는 현재 example index
        feature_indices = features_per_example[example_index]
        min_null_prediction = None
        prelim_predictions = []
        # 현재 example에 대한 모든 feature 생성합니다.
        for feature_index in feature_indices:
            # 각 featureure에 대한 모든 prediction을 가져옵니다.
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            # logit과 original context의 logit을 mapping합니다.
            offset_mapping = features[feature_index]["offset_mapping"]
            # Optional : `token_is_max_context`, 제공되는 경우 현재 기능에서 사용할 수 있는 max context가 없는 answer를 제거합니다
            # token_is_max_context = features[feature_index].get(
            #     "token_is_max_context", None
            # )

            # minimum null prediction을 업데이트 합니다.
            feature_null_score = start_logits[0] + end_logits[0]
            if (
                min_null_prediction is None
                or min_null_prediction["score"] > feature_null_score
            ):
                min_null_prediction = {
                    "offsets": (0, 0),
                    "score": feature_null_score,
                    "start_logit": start_logits[0],
                    "end_logit": end_logits[0],
                }

            # `n_best_size`보다 큰 start and end logits을 살펴봅니다.
            start_indexes = torch.argsort(start_logits, descending = True)[
                0: n_best_size +1
            ].tolist()

            end_indexes = torch.argsort(end_logits, descending = True)[0: n_best_size + 1 ].tolist()

            for start_index in start_indexes:
                for end_index in end_indexes:
                    # out-of-scope answers는 고려하지 않습니다.
                    if (
                        start_index >= len(offset_mapping)
                        or end_index >= len(offset_mapping)
                        or offset_mapping[start_index] is None
                        or offset_mapping[end_index] is None
                    ):
                        continue
                    # 길이가 < 0 또는 > max_answer_length인 answer도 고려하지 않습니다.
                    if (
                        end_index < start_index
                        or end_index - start_index + 1 > max_answer_length
                    ):
                        continue
                    # 최대 context가 없는 answer도 고려하지 않습니다.
                    # if (
                    #     token_is_max_context is not None
                    #     and not token_is_max_context.get(str(start_index), False)
                    # ):
                    #    continue
                    prelim_predictions.append(
                        {
                            "offsets": (
                                offset_mapping[start_index][0],
                                offset_mapping[end_index][1],
                            ),
                            "score": start_logits[start_index] + end_logits[end_index],
                            "start_logit": start_logits[start_index],
                            "end_logit": end_logits[end_index],
                        }
                    )

        if version_2_with_negative:
            # minimum null prediction을 추가합니다.
            prelim_predictions.append(min_null_prediction)
            null_score = min_null_prediction["score"]

        # 가장 좋은 `n_best_size` predictions만 유지합니다.
        predictions = sorted(
            prelim_predictions, key=lambda x: x["score"], reverse=True
        )[:n_best_size]

        # 낮은 점수로 인해 제거된 경우 minimum null prediction을 다시 추가합니다.
        if version_2_with_negative and not any(
            p["offsets"] == (0, 0) for p in predictions
        ):
            predictions.append(min_null_prediction)

        # offset을 사용하여 original context에서 answer text를 수집합니다.
        context = example["context"]
        for pred in predictions:
            offsets = pred.pop("offsets")
            pred["text"] = context[offsets[0] : offsets[1]]

        # rare edge case에는 null이 아닌 예측이 하나도 없으며 failure를 피하기 위해 fake prediction을 만듭니다.
        if len(predictions) == 0 or (
            len(predictions) == 1 and predictions[0]["text"] == ""
        ):

            predictions.insert(
                0, {"text": "empty", "start_logit": 0.0, "end_logit": 0.0, "score": 0.0}
            )
        if mode == "predict":
            # 모든 점수의 소프트맥스를 계산합니다(we do it with numpy to stay independent from torch/tf in this file, using the LogSumExp trick).
            scores = torch.tensor([pred.pop("score") for pred in predictions], device = torch.device('cuda'))
            exp_scores = torch.exp(scores - torch.max(scores))
            probs = exp_scores / exp_scores.sum()

            # 예측값에 확률을 포함합니다.
            for prob, pred in zip(probs, predictions):
                pred["probability"] = prob

        # best prediction을 선택합니다.
        if not version_2_with_negative:
            all_predictions[example["id"]] = predictions[0]["text"]
        else:
            # else case : 먼저 비어 있지 않은 최상의 예측을 찾아야 합니다
            i = 0
            while predictions[i]["text"] == "":
                i += 1
            best_non_null_pred = predictions[i]

            # threshold를 사용해서 null prediction을 비교합니다.
            score_diff = (
                null_score
                - best_non_null_pred["start_logit"]
                - best_non_null_pred["end_logit"]
            )
            scores_diff_json[example["id"]] = float(score_diff)  # JSON-serializable 가능
            if score_diff > null_score_diff_threshold:
                all_predictions[example["id"]] = ""
            else:
                all_predictions[example["id"]] = best_non_null_pred["text"]

        # np.float를 다시 float로 casting -> `predictions`은 JSON-serializable 가능
        all_nbest_json[example["id"]] = [
            {
                k: (
                    float(v)
                    if isinstance(v, (np.float16, np.float32, np.float64))
                    else v
                )
                for k, v in pred.items()
            }
            for pred in predictions
        ]

    if mode == "predict":
        output_dir = save_path+"/predictions"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        # output_dir이 있으면 모든 dicts를 저장합니다.
        
        prediction_file = os.path.join(
            output_dir,
            "predictions.json",
        )

        with open(prediction_file, "w", encoding="utf-8") as writer:
            writer.write(
                json.dumps(all_predictions, indent=4, ensure_ascii=False) + "\n"
            )

    return all_predictions

# Post-processing:
def post_processing_function(stage, config, id, predictions, tokenizer):
    if stage == "eval":
        examples = new_dataset
        examples = examples["validation"]
        features = examples.map(prepare_predict_features,
                                batched = True,
                                num_proc = 4,
                                remove_columns = examples.column_names,
                                fn_kwargs = {"tokenizer":tokenizer, "config":config})

    
    # Post-processing: start logits과 end logits을 original context의 정답과 match시킵니다.
    predictions = postprocess_qa_predictions(
        stage,
        examples=examples,
        features=features,
        id = id,
        predictions=predictions,
        max_answer_length=config["data"]["max_answer_length"],
        save_path=config['save_path']
    )
    # Metric을 구할 수 있도록 Format을 맞춰줍니다.
    formatted_predictions = [
        {"id": k, "prediction_text": v} for k, v in predictions.items()
    ]

    if stage == "eval":
        references = [
            {"id": ex["id"], "answers": ex["answers"]}
            for ex in examples
        ]
        return EvalPrediction(
            predictions=formatted_predictions, label_ids=references
        )

if __name__ == '__main__':
    """
    train data를 8개로 나누어 각 그룹마다 f1 score를 계산하여 정렬해 저장
    """
    with open('config.yaml') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    get_folder_name(cfg)
    
    datasets = load_from_disk(cfg["model"]["train_path"])
    
    total_datasets = len(datasets["train"])
    train_datasets = datasets["train"]
    valid_datasets = datasets["validation"]
    
    context_length = []
    for data in train_datasets:
        context = data['context']
        context_length.append(len(context))
    
    indexed_list = list(enumerate(context_length))
    sorted_indicies = [index for index, _ in sorted(indexed_list, key=lambda x: x[1])]
    
    re_dataset = []
    for i in sorted_indicies:
        convert = {key : [value] for key, value in train_datasets[i].items()}
        re_dataset.append(Dataset.from_dict(convert))
    
    combine_dataset = concatenate_datasets(re_dataset)
    
    final_dataset = DatasetDict({
        "train":combine_dataset,
        "validation": valid_datasets
    })
    
    output_dir = "./data"
    new_output_dir = "cur_train_dataset"
    new_folder_path = os.path.join(output_dir, new_output_dir)
    os.makedirs(new_folder_path)
    
    final_dataset.save_to_disk(new_folder_path)
    
    valid_datasets = datasets["validation"]
    
    # 각 부분 데이터셋의 샘플 개수 계산
    samples_per_subset = total_datasets // 8

    # 부분 데이터셋 컨테이너 생성
    subset_datasets = []

    # 원본 데이터셋을 서브셋으로 분할
    start_idx = 0
    for i in range(8):
        end_idx = start_idx + samples_per_subset
        
        # 마지막 서브셋은 나머지 샘플을 모두 포함
        if i == 7:
            end_idx = total_datasets
            
        subset = train_datasets.select(range(start_idx, end_idx))
        subset_datasets.append(subset)
    
        start_idx = end_idx
    
    model_cfg = AutoConfig.from_pretrained(cfg["model"]["model_name"])
    
    model = Model(cfg['model']['model_name'],
                  model_cfg,
                  cfg['model']['LR'], 
                  cfg['model']['loss_function'], 
                  cfg['model']['optim'], 
                  cfg['model']['scheduler'],
                  cfg)
    
    tokenizer = AutoTokenizer.from_pretrained(
        cfg['model']['model_name'], max_length = 200
    )
    
    global f1
    f1 = []
    
    for idx, subdataset in enumerate(subset_datasets):
        print(subdataset)
        print(type(subdataset))
        global new_dataset
        new_dataset = DatasetDict({"validation": subdataset})
        dataloader = Dataloader(cfg, new_dataset, tokenizer, model)
        print(new_dataset)
        trainer = pl.Trainer(accelerator='gpu', max_epochs = cfg["model"]["epoch"])
        
        trainer.test(model = model, datamodule=dataloader)
        
    indexed_list = list(enumerate(f1))
    sorted_indicies = [index for index, _ in sorted(indexed_list, key=lambda x: x[1])]
    
    re_dataset = []
    for i in sorted_indicies:
        re_dataset.append(subset_datasets[i])
        
    combine_dataset = concatenate_datasets(re_dataset)
    final_dataset = DatasetDict({
        "train":combine_dataset,
        "validation": valid_datasets
    })
    
    output_dir = "./data"
    new_output_dir = "re_train_dataset"
    new_folder_path = os.path.join(output_dir, new_output_dir)
    os.makedirs(new_folder_path)
    
    final_dataset.save_to_disk(new_folder_path)
    
        
    
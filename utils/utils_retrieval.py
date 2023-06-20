from typing import Any, Tuple, Optional, Callable, List
from datasets import DatasetDict, Features, Value, Sequence, Dataset
from utils.Retrieval import SparseRetrieval
from utils.bm25 import BM25Retrieval
from konlpy.tag import Komoran

def run_sparse_retrieval(stage, config,
    tokenize_fn: Callable[[str], List[str]],
    datasets: DatasetDict,
    data_path: str = "./data",
    context_path: str = "wikipedia_documents.json",
) -> DatasetDict:

    # Query에 맞는 Passage들을 Retrieval 합니다.
    retriever = SparseRetrieval(
        tokenize_fn=tokenize_fn, 
        data_path=data_path, 
        context_path=context_path, 
        use_normalize=config['data']['use_normalize'],
        use_sub=config['data']['use_normalize'],
        drop_duplicated_wiki = config['data']['drop_duplicated_wiki'],
        drop_less_than_50_percent_of_korean = config['data']['drop_less_than_50_percent_of_korean'],
        drop_too_long_text = config['data']['drop_too_long_text'],
        add_title_to_text = config['data']['add_title_to_text']
    )
    retriever.get_sparse_embedding()
    
    if config["data"]["use_faiss"]:
        retriever.build_faiss(num_clusters=config["data"]["num_clusters"])
        df = retriever.retrieve_faiss(
            datasets["validation"], topk=config["data"]["top_k_retrieval"]
        )
    else:
        df = retriever.retrieve(datasets["validation"], topk=config["data"]["top_k_retrieval"])
        
    # test data 에 대해선 정답이 없으므로 id question context 로만 데이터셋이 구성됩니다.
    if stage == "predict":
        f = Features(
            {
                "context": Value(dtype="string", id=None),
                "id": Value(dtype="string", id=None),
                "question": Value(dtype="string", id=None),
            }
        ) 

    # train data 에 대해선 정답이 존재하므로 id question context answer 로 데이터셋이 구성됩니다.
    elif stage == "eval":
            {
                "answers": Sequence(
                    feature={
                        "text": Value(dtype="string", id=None),
                        "answer_start": Value(dtype="int32", id=None),
                    },
                    length=-1,
                    id=None,
                ),
                "context": Value(dtype="string", id=None),
                "id": Value(dtype="string", id=None),
                "question": Value(dtype="string", id=None),
            }
    datasets = DatasetDict({"validation": Dataset.from_pandas(df, features=f)})
    return datasets

def run_bm25(stage, config,
    tokenize_fn: Callable[[str], List[str]],
    datasets: DatasetDict,
    data_path: str = "./data",
    context_path: str = "wikipedia_documents.json",
) -> DatasetDict:

    # Query에 맞는 Passage들을 Retrieval 합니다.
    retriever = BM25Retrieval(
        tokenize_fn=tokenize_fn, 
        data_path=data_path, 
        context_path=context_path, 
        stage=stage, 
        use_normalize=config['data']['use_normalize'], 
        use_sub=config['data']['use_sub'],
        use_drop_duplicated_wiki = config['data']['use_drop_duplicated_wiki'],
        use_drop_less_than_50_percent_of_korean = config['data']['use_drop_less_than_50_percent_of_korean'],
        use_drop_too_long_text = config['data']['use_drop_too_long_text'],
        use_add_title_to_text = config['data']['use_add_title_to_text']
    )
    retriever.get_bm25()
    
    df = retriever.retrieve(datasets["validation"], topk=config["data"]["top_k_retrieval"],add_ce=config["model"]["add_ce"])
        
    # test data 에 대해선 정답이 없으므로 id question context 로만 데이터셋이 구성됩니다.
    if stage == "predict":
        f = Features(
            {
                "context": Value(dtype="string", id=None),
                "id": Value(dtype="string", id=None),
                "question": Value(dtype="string", id=None),
            }
        )

    # train data 에 대해선 정답이 존재하므로 id question context answer 로 데이터셋이 구성됩니다.
    elif stage == "eval":
            {
                "answers": Sequence(
                    feature={
                        "text": Value(dtype="string", id=None),
                        "answer_start": Value(dtype="int32", id=None),
                    },
                    length=-1,
                    id=None,
                ),
                "context": Value(dtype="string", id=None),
                "id": Value(dtype="string", id=None),
                "question": Value(dtype="string", id=None),
            }
    datasets = DatasetDict({"validation": Dataset.from_pandas(df, features=f)})
    return datasets
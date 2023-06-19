import re
import unicodedata
from datasets import DatasetDict

def sub_text(
          text: str
) -> str:
        cp = re.compile('\\\\n|\*|\\n|\\|#')
        
        text = re.sub(r"[“”‘’]", "\'", text)
        text = re.sub(r"[〈<＜「≪《『]", "<", text)
        text = re.sub(r"[〉>＞」≫》』]", ">", text)
        text = cp.sub('', text)
        return text



def normalize_answer(
        dataset: DatasetDict
) -> DatasetDict:
      '''
      dataset의 answer를 반각 문자로 변환
      '''
      def normalize(example):
        example['answers']['text'][0]=unicodedata.normalize('NFKC', example['answers']['text'][0])
        return {"answers": example['answers']}
      return dataset.map(normalize)

def normalize_question(
          dataset: DatasetDict
) -> DatasetDict:
     '''
     dataset의 question을 반각 문자로 변환
     '''
     def normalize(example):
        return {"question": unicodedata.normalize('NFKC', example['question'])}
     return dataset.map(normalize)

def dataset_sub_context(
        dataset: DatasetDict,
        use_nomalize: bool
) -> DatasetDict:
    '''
    dataset 내의 context에 포함되어 있는 특수문자 제거 
    '''
    def sub(example):
        answer_start = example['answers']['answer_start'][0]
        if use_nomalize:
                new_answer_start = len(sub_text(unicodedata.normalize('NFKC', example['context'][:answer_start])))
        else:
                new_answer_start = len(sub_text(example['context'][:answer_start]))
        example['answers']['answer_start'][0] = new_answer_start
        return {"context": sub_text(example['context']), "answers": example['answers']}
    return dataset.map(sub)

def dataset_normalize_context(
          dataset: DatasetDict
) -> DatasetDict:
     '''
     dataset의 context를 반각 문자로 변환
     '''
     def normalize(example):
          return {"context": unicodedata.normalize('NFKC', example['context'])}
     return dataset.map(normalize)

def list_sub_context(
        contexts: list
) -> list:
        '''
        list 내의 context에 포함되어 있는 특수문자 제거
        '''
        for i in range(len(contexts)):
                contexts[i] = sub_text(contexts[i])
        return contexts

def list_normalize_context(
        contexts: list
) -> list:
     '''
     list의 context를 반각 문자로 변환
     '''
     for i in range(len(contexts)):
        contexts[i] = unicodedata.normalize('NFKC', contexts[i])
     return contexts

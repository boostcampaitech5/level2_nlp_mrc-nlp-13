import re
import unicodedata
from datasets import DatasetDict

import pandas as pd

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

def drop_dup(
        wiki: dict
) -> dict:
        '''
        위키피디아 text 중복행 제거
        '''
        df = pd.DataFrame(wiki).transpose()
        drop_dup_df = df.drop_duplicates(subset=['text'], keep='first')
        drop_dup_wiki = drop_dup_df.to_dict('index')
        return drop_dup_wiki

def drop_less_than_50_percent_of_korean(
        wiki: dict
) -> dict:
        '''
        위키피디아 text에서 한글 비중 50% 이하 행 제거
        '''
        df = pd.DataFrame(wiki).transpose()
        df['kor_ratio'] = df['text'].apply(count_kor_ratio)
        drop_index = df[df['kor_ratio'] < 50].index
        drop_less_than_50_df = df.drop(drop_index).drop(labels='kor_ratio', axis=1)
        new_wiki = drop_less_than_50_df.to_dict('index')
        
        return new_wiki

        
def count_kor_ratio(
        text: str
) -> float:
        '''
        위키피디아 text에서 한글 비중 계산 함수
        '''

        processed_text = re.sub(r'[\n\s]','', text) #띄어쓰기 붙이기, \n 전처리
        p = re.compile('[가-힣]')
        cnt_kor = len(p.findall(processed_text))
        ratio = cnt_kor/len(processed_text)*100
        
        return ratio


def drop_too_long_text(
        wiki: dict
) -> dict:
        '''
        위키피디아 text에서 상위 1% 길이의 text 제거
        '''
        df = pd.DataFrame(wiki).transpose()
        df['length'] = df['text'].apply(lambda x:len(re.sub(r'[\n\s]','', x)))
        df['length_qcut'] = pd.qcut(df['length'],100, labels=range(100))
        drop_index = df[df['length_qcut'] == 99].index
        drop_too_long_df = df.drop(drop_index).drop(labels=['length', 'length_qcut'])
        new_wiki = drop_too_long_df.to_dict('index')
        
        return new_wiki
        
        

        
        

        
        
        
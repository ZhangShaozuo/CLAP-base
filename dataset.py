
from collections import Counter
from dataclasses import dataclass
from datasets import load_dataset
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
from transformers import Wav2Vec2Processor, AutoTokenizer
from typing import Optional, Dict, List, Union, Any
import os
import numpy as np
import pandas as pd
import re
import torch
import torchaudio
class LADataset_Base:
    def __init__(self, raw_path, out_path, l_ckpt, a_ckpt, max_wav_length, device) -> None:
        self.raw_path = raw_path
        self.out_path = out_path
        
        self.max_wav_length = max_wav_length
        self.processor = Wav2Vec2Processor.from_pretrained(a_ckpt)
        self.target_sampling_rate = self.processor.feature_extractor.sampling_rate
        
        self.tokenizer = AutoTokenizer.from_pretrained(l_ckpt)
        self.device = device

        self.wav_path_col = 'wav_path'
        self.text_path_col = 'txt_path'
        self.text_col = 'transcr'
        self.labels_col = 'emotions'
        self.label_col = 'label'
        self.stat_col = 'token_count'
        self.batch_process = 1

    def _get_data(self) -> dict:
        raise NotImplementedError
    
    def _preprocess(self, examples):
        raise NotImplementedError
    
    def _wav_path2arr(self, path):
        speech_array, sampling_rate = torchaudio.load(path)
        resampler = torchaudio.transforms.Resample(sampling_rate, self.target_sampling_rate)
        speech = resampler(speech_array).squeeze().numpy()[:self.max_wav_length]
        return speech
    
    def _txt_path2arr(self, examples):
        text_list = []
        for i in range(len(examples[self.text_path_col])):
            path, label = examples[self.text_path_col][i], examples[self.labels_col][i]
            with open(path, 'r') as f:
                line = f.readlines()[0]
            text_list.append(f'{line} Emotion: {label}')
        return text_list

class LADataset_IEMOCap(LADataset_Base):
    def __init__(self, raw_path, out_path, l_ckpt, a_ckpt, max_wav_length, device) -> None:
        super().__init__(raw_path, out_path, l_ckpt, a_ckpt, max_wav_length, device)
    
    def _postprocess_df(self, df):
        '''Emotions are annotated by 3 experts, this function is to
        1) remove duplicate id
        2) concat emotion strings from 3 experts for the same id '''
        ids, df_l = pd.unique(df['id']), []
        for id in ids:
            df_l.append({'id': id, self.labels_col: '; '.join(df[df['id']==id][self.labels_col])})
        return pd.DataFrame(df_l)
    
    def _extract_majority(self, x):

        s = str(x).split('; ')
        s[-1] = s[-1].split('; (')[0]
        most_fre = [m[0] for m in Counter(s).most_common()]
        most_fre = [m for m in most_fre if m[0] != '(']
        if len(most_fre)>1 and ('Other' in most_fre):
            most_fre.remove('Other')
        if len(most_fre)>1 and ('Neutral state' in most_fre):
            most_fre.remove('Neutral state')
        return ','.join(sorted(most_fre))

    def _extract_labels(self, x):
        s = str(x).split(' :')[1:]
        s[-1] = s[-1].split('; ()')[0].strip()
        return ' '.join(s)
    
    def _preprocess(self, examples):
        speech_list = [self._wav_path2arr(path) for path in examples[self.wav_path_col]]
        # text_list = self._txt_path2arr(examples)
        text_list = [examples[self.text_col][i] + '. All emotions: '+ examples['emotions'][i] for i in range(len(examples['id']))]
        # target_list = [self.label_list.index(label) for label in examples[self.labels_col]]
        result = self.processor(speech_list, sampling_rate=self.target_sampling_rate, max_length = self.max_wav_length)

        result["input_ids"] = self.tokenizer(text_list)["input_ids"]
        result["labels"] = examples[self.label_col]
        return result
    
    def _get_data(self) -> pd.DataFrame:
        dir_paths = [p for p in Path(self.raw_path).iterdir() if p.is_dir()]
        wav_df, transcr_df, emt_df = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        for ses in tqdm(dir_paths):
            '''wav_path --> pd.DataFrame'''
            wav_paths = os.path.join(ses,'wav')
            dialog_paths = [p for p in Path(wav_paths).iterdir() if p.is_dir()]
            for dialog in dialog_paths:
                wav_df_t = pd.DataFrame(dialog.glob('**/*.wav'), columns=[self.wav_path_col])
                wav_df_t['id'] = wav_df_t.apply(lambda x: str(x[self.wav_path_col]).split('/')[-1].split('.')[0].strip(), axis=1)
                wav_df = pd.concat([wav_df, wav_df_t], ignore_index = True)
            
            '''transcrptions --> pd.DataFrame'''
            txt_paths = os.path.join(ses, 'transcriptions')
            for transcr in Path(txt_paths).glob('**/*.txt'):
                with open(transcr, 'r') as f:
                    lines = [line for line in f.readlines() if ('Ses' in line) and ('XX' not in line)]
                    transcr_df_t = pd.DataFrame(lines, columns = ['_raw'])
                    transcr_df_t['id'] = transcr_df_t.apply(lambda x: str(x['_raw'].split(' ')[0]).strip(), axis = 1)
                    transcr_df_t[self.text_col] = transcr_df_t.apply(lambda x: str(x['_raw']).split(': ')[1].strip(), axis = 1)
                    del transcr_df_t['_raw']
                    transcr_df = pd.concat([transcr_df, transcr_df_t], ignore_index = True)
            '''Emotions --> pd.DataFrame'''
            emt_paths = os.path.join(ses, 'Categorical')
            
            for emts in Path(emt_paths).glob('**/*.txt'):
                with open(emts, 'r') as f:
                    lines = f.readlines()
                    emt_df_t = pd.DataFrame(lines, columns=['_raw'])
                    emt_df_t['id'] = emt_df_t.apply(lambda x: str(x['_raw'].split(' ')[0]), axis = 1)
                    emt_df_t[self.labels_col] = emt_df_t.apply(
                        lambda x: self._extract_labels(x['_raw']), 
                        axis = 1)
                    del emt_df_t['_raw']
                    emt_df = pd.concat([emt_df, emt_df_t], ignore_index = True)
            
            emt_df = self._postprocess_df(emt_df)

        assert wav_df.size == transcr_df.size == emt_df.size
        meta = wav_df.join(transcr_df.set_index('id'), on = 'id')
        data = meta.join(emt_df.set_index('id'), on = 'id')
        
        data[self.label_col] = data.apply(lambda x: self._extract_majority(x[self.labels_col]), axis = 1)
        data[self.stat_col] = data.apply(lambda x: len(x[self.text_col].split(' ')), axis = 1)

        data = data.loc[data[self.stat_col] > 2]
        return data
    
    def save_data(self):
        df = self._get_data()
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=101)
        train_df = train_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)
        train_df.to_csv(f"{self.out_path}/train.csv", sep="\t", encoding="utf-8", index=False)
        test_df.to_csv(f"{self.out_path}/test.csv", sep="\t", encoding="utf-8", index=False)
        data_files = {
                "train": f"{self.out_path}/train.csv", 
                "test": f"{self.out_path}/test.csv"}
        dataset = load_dataset("csv", data_files=data_files, delimiter="\t")
        train_dataset, test_dataset = dataset["train"], dataset["test"]
        train_dataset = train_dataset.map(
            self._preprocess,
            batch_size=self.batch_process,
            batched=True,
            num_proc=4
        )
        test_dataset = test_dataset.map(
            self._preprocess,
            batch_size=self.batch_process,
            batched=True,
            num_proc=4
        )
        train_dataset.save_to_disk(f'{self.out_path}/train_dataset')
        test_dataset.save_to_disk(f'{self.out_path}/test_dataset')
    

class LADataset_ESD(LADataset_Base):
    def __init__(self, raw_path, out_path, l_ckpt, a_ckpt, max_wav_length, device) -> None:
        super().__init__(raw_path, out_path, l_ckpt, a_ckpt, max_wav_length, device)
        self.label_list = ["Angry","Happy","Neutral","Sad","Surprise"]
    
    def _get_data(self) -> dict:
        data = {'train':[], 'test':[], 'evaluation':[]}
        dir_paths = [p for p in Path(self.raw_path).iterdir() if p.is_dir()]

        for spker_id in tqdm(dir_paths):
            emts_p =  [p for p in Path(spker_id).iterdir() if p.is_dir()]
            for emt in tqdm(emts_p):
                splits_p = [p for p in Path(emt).iterdir() if p.is_dir()]
                for split in splits_p:
                    for aud in Path(split).glob("**/*.wav"):
                        id = str(aud).split('/')[-1].split('.')[0]
                        key = str(aud).split('/')[-2]
                        label = str(aud).split('/')[-3]
                        data[key].append({
                            "id": id,
                            self.wav_path_col: str(aud),
                            self.text_path_col: os.path.join(split, key, id+'.lab'),
                            self.labels_col: label
                        })
        return data
    
    def _preprocess(self, examples):
        speech_list = [self._wav_path2arr(path) for path in examples[self.wav_path_col]]
        text_list = self._txt_path2arr(examples)
        # text_list = [examples[self.text_col][i] + '. All emotions: '+ examples['emotions'][i] for i in range(len(examples['id']))]
        target_list = [self.label_list.index(label) for label in examples[self.labels_col]]
        result = self.processor(speech_list, sampling_rate=self.target_sampling_rate, max_length = self.max_wav_length)

        result["input_ids"] = self.tokenizer(text_list)["input_ids"]
        # result["labels"] = examples[self.label_col]
        result["labels"] = target_list
        return result
    
    def save_data(self):
        data = self._get_data()
        train_data, eval_data, test_data = data['train'], data['evaluation'], data['test']
        train_df, eval_df, test_df = pd.DataFrame(train_data), pd.DataFrame(eval_data), pd.DataFrame(test_data)
        train_df.to_csv(f"{self.out_path}/train.csv", sep="\t", encoding="utf-8", index=False)
        eval_df.to_csv(f"{self.out_path}/eval.csv", sep="\t", encoding="utf-8", index=False)
        test_df.to_csv(f"{self.out_path}/test.csv", sep="\t", encoding="utf-8", index=False)
        data_files = {
                "train": f"{self.out_path}/train.csv", 
                "validation": f"{self.out_path}/test.csv",
                "test": f"{self.out_path}/eval.csv"}
        dataset = load_dataset("csv", data_files=data_files, delimiter="\t")
        train_dataset, eval_dataset, test_dataset = dataset["train"], dataset["validation"], dataset["test"]
        self.label_list = train_dataset.unique("emotion")
        train_dataset = train_dataset.map(
            self._preprocess,
            batched=True,
            num_proc=4
        )
        eval_dataset = eval_dataset.map(
            self._preprocess,
            batch_size=1,
            batched=True,
            num_proc=4
        )
        test_dataset = test_dataset.map(
            self._preprocess,
            batch_size=1,
            batched=True,
            num_proc=4
        )
        train_dataset.save_to_disk(f'{self.out_path}/train_dataset')
        eval_dataset.save_to_disk(f'{self.out_path}/evaluation_dataset')
        test_dataset.save_to_disk(f'{self.out_path}/test_dataset')

class LADataset_MELD(LADataset_Base):
    def __init__(self, raw_path, out_path, l_ckpt, a_ckpt, max_wav_length, device) -> None:
        super().__init__(raw_path, out_path, l_ckpt, a_ckpt, max_wav_length, device)

    def _get_data(self) -> dict:
        train_df = pd.read_csv(f'{self.raw_path}/train_sent_emo.csv')
        dev_df = pd.read_csv(f'{self.raw_path}/dev_sent_emo.csv')
        train_df = self._get_data_sub(train_df)
        dev_df = self._get_data_sub(dev_df)
        return train_df, dev_df
    
    def _get_data_sub(self, df):
        df[self.wav_path_col] = df.apply(
            lambda x: self._extract_wav_path(x['Dialogue_ID'], x['Utterance_ID']), axis=1)
        df['wav_exist'] = df[self.wav_path_col].apply(lambda x: os.path.exists(x))
        df = df[df['wav_exist'] == True]
        df['Utterance'] = df['Utterance'].apply(lambda x: re.sub(r'[^\w\s]', '', x))
        df['token_num'] = df['Utterance'].apply(lambda x: len(x.split(' ')))
        df = df[df['token_num'] > 2]
        df = df[['Utterance', 'Emotion', self.wav_path_col]]
        df = df.rename(columns={'Utterance': self.text_col, 'Emotion': self.label_col})
        return df
    
    def _extract_wav_path(self, dialog_id, utterance_id):
        return os.path.join(self.raw_path, 'train_wav', 'dia{}_utt{}.wav'.format(dialog_id, utterance_id))
    
    def _preprocess(self, examples):
        speech_list = [self._wav_path2arr(path) for path in examples[self.wav_path_col]]
        text_list = [examples[self.text_col][i] + '. Emotion: '+ examples[self.label_col][i] for i in range(len(examples[self.label_col]))]
        result = self.processor(speech_list, sampling_rate=self.target_sampling_rate)

        result["input_ids"] = self.tokenizer(text_list)["input_ids"]
        result["labels"] = examples[self.label_col]
        return result
    
    def save_data(self):
        # train_df, dev_df = self._get_data()
        # train_df.to_csv(f"{self.out_path}/train.csv", sep="\t", encoding="utf-8", index=False)
        # dev_df.to_csv(f"{self.out_path}/dev.csv", sep="\t", encoding="utf-8", index=False)
        data_files = {
                "train": f"{self.out_path}/train_mixed.csv", 
                "dev": f"{self.out_path}/dev.csv"}
        dataset = load_dataset("csv", data_files=data_files, delimiter="\t")
        train_dataset, dev_dataset = dataset["train"], dataset["dev"]
        self.label_list = train_dataset.unique(self.label_col)
        train_dataset = train_dataset.map(
            self._preprocess,
            batched=True,
            num_proc=4)
        dev_dataset = dev_dataset.map(
            self._preprocess,
            batched=True,
            num_proc=4)
        train_dataset.save_to_disk(f'{self.out_path}/train_dataset')
        dev_dataset.save_to_disk(f'{self.out_path}/dev_dataset')

    def save_data_meta(self):
        df_1 = pd.read_csv(f'{self.raw_path}/train.csv', sep='\t')
        df_2 = pd.read_csv(f'{self.out_path}/IEMOCap.csv', sep='\t')
        train_df = pd.concat([df_1, df_2], axis = 0).reset_index(drop=True)
        train_df.to_csv(f"{self.out_path}/train_mixed.csv", sep="\t", encoding="utf-8", index=False)
        
class LADataset_ZSCLS(LADataset_MELD):
    def __init__(self, raw_path, out_path, l_ckpt, a_ckpt, max_wav_length, device) -> None:
        super().__init__(raw_path, out_path, l_ckpt, a_ckpt, max_wav_length, device)
    
    def _preprocess(self, examples):
        speech_list = [self._wav_path2arr(path) for path in examples[self.wav_path_col]]
        result = self.processor(speech_list, sampling_rate=self.target_sampling_rate)
        
        # text_list = [examples[self.text_col][i] + '. Emotion: '+ examples[self.label_col][i] for i in range(len(examples[self.label_col]))]
        for idx in range(len(self.label_list)):
            text_list = [examples[self.text_col][i] + f'. Emotion: {self.label_list[idx]}' for i in range(len(examples[self.label_col]))]
            result[f'input_ids_{idx}'] = self.tokenizer(text_list)["input_ids"]
        # result["input_ids"] = self.tokenizer(text_list)["input_ids"]
        
        result["labels"] = [self.label_list.index(e) for e in examples[self.label_col]]
        # breakpoint()
        return result
    
    def save_data(self):
        data_files = {'dev': f"{self.out_path}/dev.csv"}
        dataset = load_dataset('csv', data_files=data_files, delimiter="\t")['dev']
        self.label_list = dataset.unique(self.label_col)

        # self._preprocess(dataset[:100])
        dataset = dataset.map(
            self._preprocess,
            batched=True,
            num_proc=4)

        dataset.save_to_disk(f'{self.out_path}/dev_PN_dataset')
@dataclass
class DataCollatorPNWithPadding:
    '''
    Use case: wav2vec2 audio inputs. Process: 1)pad input_values 2)retrieve attention masks 3)save to device gpu or cpu
    '''
    processor: Wav2Vec2Processor
    tokenizer: AutoTokenizer
    padding: Union[bool, str] = True
    num_labels: Optional[int] = None
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None
    device: Optional[torch.device] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        a_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [feature["labels"] for feature in features]
        batch = self.processor.pad(
            a_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt"
        )
        for i in range(self.num_labels):
            l_features = [{"input_ids": feature[f"input_ids_{i}"]} for feature in features]
            l_batch = self.tokenizer.pad(
                l_features,
                padding = self.padding,
                max_length = self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors="pt"
            )
            batch[f'input_ids_{i}'] = l_batch["input_ids"]
            batch[f'l_attn_mask_{i}'] = torch.where(batch[f"input_ids_{i}"] == 0, 0, 1)
        
        batch['a_attn_mask'] = torch.where(batch["input_values"] == 0, 0, 1)
        batch['labels'] = label_features
        return batch
@dataclass
class DataCollatorCTCWithPadding:
    '''
    Use case: wav2vec2 audio inputs. Process: 1)pad input_values 2)retrieve attention masks 3)save to device gpu or cpu
    '''
    processor: Wav2Vec2Processor
    tokenizer: AutoTokenizer
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None
    device: Optional[torch.device] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        a_features = [{"input_values": feature["input_values"]} for feature in features]
        l_features = [{"input_ids": feature["input_ids"]} for feature in features]
        label_features = [feature["labels"] for feature in features]

        # d_type = torch.long if isinstance(label_features[0], int) else torch.float

        batch = self.processor.pad(
            a_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt"
        )
        l_batch = self.tokenizer.pad(
            l_features,
            padding = self.padding,
            max_length = self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt"
        )
        batch['input_ids'] = l_batch["input_ids"]
        batch['a_attn_mask'] = torch.where(batch["input_values"]==0, 0, 1)
        batch['l_attn_mask'] = torch.where(batch["input_ids"]==0,0,1)
        batch['labels'] = label_features
        return batch
    
def main_IEMOCap():
    dataset = LADataset_IEMOCap(raw_path = 'datasets/IEMOCap/raw',
                        out_path = 'datasets/IEMOCap/preprocessed_data',
                        l_ckpt = 'bert-base-uncased',
                        a_ckpt = 'facebook/wav2vec2-base-960h',
                        max_wav_length = 300000,
                        device = torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    dataset.save_data()

def main_MELD():
    dataset = LADataset_MELD(raw_path = 'datasets/MELD/raw',
                             out_path = 'datasets/MELD/preprocessed_data',
                             l_ckpt = 'bert-base-uncased',
                             a_ckpt = 'facebook/wav2vec2-base-960h',
                             max_wav_length = 300000,
                             device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    dataset.save_data()

def main_Mixed():
    dataset = LADataset_MELD(
                             raw_path = 'datasets/Mixed',
                             out_path = 'datasets/Mixed',
                             l_ckpt = 'bert-base-uncased',
                             a_ckpt = 'facebook/wav2vec2-base-960h',
                             max_wav_length = 250000,
                             device = torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    dataset.save_data()

def main_pos_neg():
    dataset = LADataset_ZSCLS(raw_path = 'datasets/Mixed',
                            out_path = 'datasets/Mixed',
                            l_ckpt = 'bert-base-uncased',
                            a_ckpt = 'facebook/wav2vec2-base-960h',
                            max_wav_length = 250000,
                            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    dataset.save_data()
def test():
    path = 'datasets/MELD/raw/train_wav/dia0_utt0.wav'
    audio, sampling_rate = torchaudio.load(path)
    print(audio.shape)
    breakpoint()

if __name__ == '__main__':
    # main_MELD()
    # main_Mixed()
    # test()
    main_pos_neg()
    
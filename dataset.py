from typing import Optional

from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
import pandas as pd

def collate_EAP(xs):
    clean, corrupted, labels = zip(*xs)
    clean = list(clean)
    corrupted = list(corrupted)
    return clean, corrupted, labels

class EAPDataset(Dataset):
    def __init__(self, filepath:str, task:str='greater-than'):
        self.task = task
        self.df = pd.read_csv(filepath)

    def __len__(self):
        return len(self.df)
    
    def shuffle(self):
        self.df = self.df.sample(frac=1)

    def head(self, n: int):
        self.df = self.df.head(n)
    
    def __getitem__(self, index):
        row = self.df.iloc[index]
        if self.task == 'greater-than':
            return row['clean'], row['corrupted'], row['correct_idx']
        elif self.task == 'ioi':
            return row['clean'], row['corrupted'], [row['correct_idx'], row['incorrect_idx']]
        else:
            raise ValueError(f'Got invalid task: {self.task}')
    
    def to_dataloader(self, batch_size: int):
        return DataLoader(self, batch_size=batch_size, collate_fn=collate_EAP)
    
class HFEAPDataset(Dataset):
    def __init__(self, url:str, tokenizer, split:str="train", task:str='ioi', num_examples:Optional[int]=None):
        self.task = task
        self.tokenizer = tokenizer
        if task == 'mcqa':
            self.dataset = load_dataset(url, '2_answer_choices', split=split)
        else:
            self.dataset = load_dataset(url, split=split)
        self.dataset = self.filter_dataset()
        #self.dataset = self.shuffle()
        if num_examples:
            self.dataset = self.head(num_examples)

    def __len__(self):
        return len(self.dataset)
    
    def shuffle(self):
        return self.dataset.shuffle()
    
    def head(self, n: int):
        return [self.dataset[i] for i in range(n)]

    def filter_dataset(self):
        if self.task == 'ioi':
            filtered_dataset = self.dataset.filter(
                lambda x: len(self.tokenizer(f" {x['metadata']['indirect_object']}", add_special_tokens=False).input_ids) == 
                          len(self.tokenizer(f" {x['metadata']['subject']}", add_special_tokens=False).input_ids)
            )
            return filtered_dataset
        elif self.task == 'mcqa':
            return self.dataset
        else:
            raise ValueError(f"Unrecognized task: {self.task}")
    
    def __getitem__(self, index):
        row = self.dataset[index]
        if self.task == 'ioi':
            counterfactual_col = 's2_io_flip_counterfactual' if True else 'abc_counterfactual'
            correct_idx = self.tokenizer(f" {row['metadata']['indirect_object']}", add_special_tokens=False).input_ids[0]
            incorrect_idx = self.tokenizer(f" {row['metadata']['subject']}", add_special_tokens=False).input_ids[0]
            return row['prompt'], row[counterfactual_col]['prompt'], [correct_idx, incorrect_idx]
        elif self.task == 'mcqa':
            clean_prompt = row["prompt"]
            correct_idx = self.tokenizer(row["choices"]["label"][row["answerKey"]], add_special_tokens=False).input_ids[0]
            counterfactual_cols = [k for k in list(row.keys()) if "_counterfactual" in k]
            counterfactual_col = row[counterfactual_cols[index % len(counterfactual_cols)]]
            corrupted_prompt = counterfactual_col["prompt"]
            incorrect_ans = str(counterfactual_col["choices"]["label"][counterfactual_col["answerKey"]])
            incorrect_idx = self.tokenizer(incorrect_ans, add_special_tokens=False).input_ids[0]
            return clean_prompt, corrupted_prompt, [correct_idx, incorrect_idx]
    
    def to_dataloader(self, batch_size: int):
        return DataLoader(self, batch_size=batch_size, collate_fn=collate_EAP)
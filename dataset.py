from typing import Optional

from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
import pandas as pd
import random

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
        elif self.task == 'ewok':
            return row['Context1'], row['Context2'], [row['Target1'], row['Target2']]
        else:
            raise ValueError(f'Got invalid task: {self.task}')
    
    def to_dataloader(self, batch_size: int):
        return DataLoader(self, batch_size=batch_size, collate_fn=collate_EAP)
    
class HFEAPDataset(Dataset):
    def __init__(self, url:str, tokenizer, split:str="train", task:str='ioi', num_examples:Optional[int]=None,
                 control:Optional[bool]=False, counterfactual_type:Optional[str]=None,
                 example_domain:Optional[str]=None):      
        self.task = task
        self.tokenizer = tokenizer
        self.control = control

        if task == 'mcqa':
            self.dataset = load_dataset(url, '2_answer_choices', split=split)
            if counterfactual_type is None:
                self.counterfactual_type = "symbol_counterfactual"
            else:
                self.counterfactual_type = counterfactual_type

        elif task == 'ewok':
            self.dataset = load_dataset(url, split="test")
            if example_domain is not None:
                self.example_domain = example_domain
            else:
                self.example_domain = "social-properties"
        
        else:
            self.dataset = load_dataset(url, split=split)
        
        self.dataset = self.filter_dataset()
        #self.dataset = self.shuffle()
        if num_examples:
            self.dataset = self.head(num_examples)
        
        # for when `control is True`
        self.answer_map = {}

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
                          len(self.tokenizer(f" {x['metadata']['subject']}", add_special_tokens=False).input_ids) and
                          len(self.tokenizer(f" {x['metadata']['indirect_object']}", add_special_tokens=False).input_ids) ==
                          len(self.tokenizer(f" {x['metadata']['random_c']}", add_special_tokens=False).input_ids)
            )
            return filtered_dataset
        elif self.task == 'mcqa':
            filtered_dataset = self.dataset.filter(
                lambda x: len(self.tokenizer(x["choices"]["label"][x["answerKey"]], add_special_tokens=False).input_ids) ==
                          len(self.tokenizer(str(x[self.counterfactual_type]["choices"]["label"][x[self.counterfactual_type]["answerKey"]]),
                                             add_special_tokens=False).input_ids)
            )
            return filtered_dataset
        elif self.task == 'ewok':
            filtered_dataset = self.dataset.filter(
                lambda x: len(self.tokenizer(x["Target1"], add_special_tokens=False).input_ids) ==
                          len(self.tokenizer(x["Target2"], add_special_tokens=False).input_ids) and
                          x["Domain"] == self.example_domain
            )
        else:
            raise ValueError(f"Unrecognized task: {self.task}")
    
    def __getitem__(self, index):
        row = self.dataset[index]
        if self.task == 'ioi':
            counterfactual_col = 's2_io_flip_cf' if True else 'abc_cf'
            correct_idx = self.tokenizer(f" {row['metadata']['indirect_object']}", add_special_tokens=False).input_ids[0]
            incorrect_idx = self.tokenizer(f" {row['metadata']['subject']}", add_special_tokens=False).input_ids[0]
            if self.control:
                random.seed(index)
                if correct_idx in self.answer_map:
                    correct_idx = self.answer_map[correct_idx]
                else:
                    self.answer_map[correct_idx] = random.randint(1000, self.tokenizer.vocab_size-1000)
                    correct_idx = self.answer_map[correct_idx]
                if incorrect_idx in self.answer_map:
                    incorrect_idx = self.answer_map[incorrect_idx]
                else:
                    self.answer_map[incorrect_idx] = random.randint(1000, self.tokenizer.vocab_size-1000)
                    incorrect_idx = self.answer_map[incorrect_idx]
            return row['text'], row['counterfactuals'][counterfactual_col], [correct_idx, incorrect_idx]
        
        elif self.task == 'mcqa':
            clean_prompt = row["prompt"]
            correct_idx = self.tokenizer(row["choices"]["label"][row["answerKey"]], add_special_tokens=False).input_ids[0]
            # counterfactual_cols = [k for k in list(row.keys()) if "_counterfactual" in k]
            counterfactual_col = row[self.counterfactual_type]
            corrupted_prompt = counterfactual_col["prompt"]
            incorrect_ans = str(counterfactual_col["choices"]["label"][counterfactual_col["answerKey"]])
            incorrect_idx = self.tokenizer(incorrect_ans, add_special_tokens=False).input_ids[0]
            if self.control:
                random.seed(index)
                if correct_idx in self.answer_map:
                    correct_idx = self.answer_map[correct_idx]
                else:
                    self.answer_map[correct_idx] = random.randint(1000, self.tokenizer.vocab_size-1000)
                    correct_idx = self.answer_map[correct_idx]
                if incorrect_idx in self.answer_map:
                    incorrect_idx = self.answer_map[incorrect_idx]
                else:
                    self.answer_map[incorrect_idx] = random.randint(1000, self.tokenizer.vocab_size-1000)
                    incorrect_idx = self.answer_map[incorrect_idx]
            return clean_prompt, corrupted_prompt, [correct_idx, incorrect_idx]

        elif self.task == 'ewok':
            clean_prompt = row["Context1"]
            counterfactual_prompt = row["Context2"]
            correct_idxs = self.tokenizer(row["Target1"], add_special_tokens=False).input_ids
            incorrect_idxs = self.tokenizer(row["Target2"], add_special_tokens=False).input_ids
            if self.control:
                raise NotImplementedError("Error: controls are not implemented for multi-token outputs.")
            return clean_prompt, counterfactual_prompt, [correct_idxs, incorrect_idxs]

    def to_dataloader(self, batch_size: int):
        return DataLoader(self, batch_size=batch_size, collate_fn=collate_EAP)
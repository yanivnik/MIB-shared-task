from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
import pandas as pd

def collate_EAP(xs):
    clean, corrupted, labels = zip(*xs)
    clean = list(clean)
    corrupted = list(corrupted)
    return clean, corrupted, labels

class EAPDataset(Dataset):
    def __init__(self, filepath, task='greater-than'):
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
    def __init__(self, url, tokenizer, split="train", task='ioi', num_examples=None):
        self.task = task
        self.tokenizer = tokenizer
        self.dataset = load_dataset(url, split=split)
        self.dataset = self.filter_dataset()
        self.dataset = self.shuffle()
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
                lambda x: len(self.tokenizer(f" {x['metadata']['IO']}", add_special_tokens=False).input_ids) == 1 and \
                          len(self.tokenizer(f" {x['metadata']['S']}", add_special_tokens=False).input_ids) == 1
            )
            return filtered_dataset
        else:
            raise ValueError(f"Unrecognized task: {self.task}")
    
    def __getitem__(self, index):
        row = self.dataset[index]
        if self.task == 'ioi':
            clean_prompt = " ".join(row["text"].split()[:-1])
            IO, S = row['metadata']["IO"], row['metadata']["S"]
            prompt_parts = clean_prompt.split(S)
            corrupted_prompt = f"{S.join(prompt_parts[:-1])}{IO}{prompt_parts[-1]}"
            correct_idx = self.tokenizer(f" {row['metadata']['IO']}", add_special_tokens=False).input_ids[0]
            incorrect_idx = self.tokenizer(f" {row['metadata']['S']}", add_special_tokens=False).input_ids[0]
            return clean_prompt, corrupted_prompt, [correct_idx, incorrect_idx]
    
    def to_dataloader(self, batch_size: int):
        return DataLoader(self, batch_size=batch_size, collate_fn=collate_EAP)
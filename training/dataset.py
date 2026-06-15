import torch
from torch.utils.data import Dataset
from typing import Optional
from ..tokenizers import SemanticTokenizer


class TextDataset(Dataset):
    def __init__(self, file_path: str, tokenizer: SemanticTokenizer, seq_len: int = 1024):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        
        with open(file_path, 'r', encoding='utf-8') as f:
            self.text = f.read()
        
        self.tokens = self.tokenizer.encode(self.text).ids
        self.num_sequences = len(self.tokens) // seq_len
    
    def __len__(self):
        return self.num_sequences
    
    def __getitem__(self, idx):
        start = idx * self.seq_len
        end = start + self.seq_len + 1
        
        if end > len(self.tokens):
            end = len(self.tokens)
            start = max(0, end - self.seq_len - 1)
        
        seq = self.tokens[start:end]
        
        if len(seq) < self.seq_len + 1:
            seq = seq + [self.tokenizer.special_tokens["<PAD>"]] * (self.seq_len + 1 - len(seq))
        
        input_ids = torch.tensor(seq[:-1], dtype=torch.long)
        target_ids = torch.tensor(seq[1:], dtype=torch.long)
        
        return input_ids, target_ids

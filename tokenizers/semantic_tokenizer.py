import json
from pathlib import Path
from typing import Dict, List, Tuple
from collections import Counter
import numpy as np


class SemanticTokenizer:
    def __init__(self, vocab_size: int = 8000):
        self.vocab_size = vocab_size
        self.special_tokens = {
            "<PAD>": 0,
            "<UNK>": 1,
            "<START>": 2,
            "<STOP>": 3,
        }
        
        self.byte_seq_to_id: Dict[bytes, int] = {}
        self.id_to_byte_seq: Dict[int, bytes] = {}
        self.next_id = len(self.special_tokens)
        self.vocab_version = 1
    
    def train(self, file_path: str):
        with open(file_path, 'rb') as f:
            data = f.read()
        
        byte_counts = Counter(data)
        
        for byte_val, freq in byte_counts.most_common():
            if self.next_id >= self.vocab_size:
                break
            byte_bytes = bytes([byte_val])
            if byte_bytes not in self.byte_seq_to_id:
                self._add_token(byte_bytes, freq)
        
        self._learn_merges(data)
    
    def _add_token(self, byte_seq: bytes, freq: int):
        if byte_seq not in self.byte_seq_to_id:
            self.byte_seq_to_id[byte_seq] = self.next_id
            self.id_to_byte_seq[self.next_id] = byte_seq
            self.next_id += 1
    
    def _learn_merges(self, data: bytes):
        token_ids = [self.byte_seq_to_id.get(bytes([b]), self.special_tokens["<UNK>"]) for b in data]
        token_ids = np.array(token_ids, dtype=np.int32)
        
        iteration = 0
        max_iterations = self.vocab_size - self.next_id
        
        while self.next_id < self.vocab_size and iteration < max_iterations:
            iteration += 1
            
            pair_frequencies = {}
            for i in range(len(token_ids) - 1):
                pair = (token_ids[i], token_ids[i + 1])
                pair_frequencies[pair] = pair_frequencies.get(pair, 0) + 1
            
            if not pair_frequencies:
                break
            
            best_pair, freq = max(pair_frequencies.items(), key=lambda x: x[1])
            
            if freq < 2:
                break
            
            id1, id2 = best_pair
            merged_bytes = self.id_to_byte_seq[id1] + self.id_to_byte_seq[id2]
            
            if merged_bytes in self.byte_seq_to_id:
                break
            
            new_token_id = self.next_id
            self._add_token(merged_bytes, freq)
            
            new_token_ids = []
            i = 0
            while i < len(token_ids):
                if i < len(token_ids) - 1 and token_ids[i] == id1 and token_ids[i + 1] == id2:
                    new_token_ids.append(new_token_id)
                    i += 2
                else:
                    new_token_ids.append(token_ids[i])
                    i += 1
            token_ids = np.array(new_token_ids, dtype=np.int32)
            
            if iteration % 100 == 0:
                print(f"Iteration {iteration}: vocab={self.next_id}")
    
    def encode(self, text: str):
        byte_seq = text.encode('utf-8')
        token_ids = []
        i = 0
        
        while i < len(byte_seq):
            found = False
            for length in range(min(32, len(byte_seq) - i), 0, -1):
                sub_seq = byte_seq[i:i+length]
                if sub_seq in self.byte_seq_to_id:
                    token_ids.append(self.byte_seq_to_id[sub_seq])
                    i += length
                    found = True
                    break
            
            if not found:
                token_ids.append(self.special_tokens["<UNK>"])
                i += 1
        
        return EncodingResult(token_ids)
    
    def decode(self, token_ids: List[int]) -> str:
        byte_seq = b''
        for tid in token_ids:
            if tid in self.id_to_byte_seq:
                byte_seq += self.id_to_byte_seq[tid]
            elif tid == self.special_tokens["<UNK>"]:
                byte_seq += b'?'
        return byte_seq.decode('utf-8', errors='ignore')
    
    def get_vocab_size(self) -> int:
        return self.next_id
    
    def save(self, filepath: str):
        data = {
            'vocab_size': self.vocab_size,
            'special_tokens': self.special_tokens,
            'byte_seq_to_id': {k.decode('utf-8', errors='ignore'): v for k, v in self.byte_seq_to_id.items()},
            'id_to_byte_seq': {k: v.decode('utf-8', errors='ignore') for k, v in self.id_to_byte_seq.items()},
            'vocab_version': self.vocab_version
        }
        with open(filepath, 'w') as f:
            json.dump(data, f)
    
    def load(self, filepath: str):
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.vocab_size = data['vocab_size']
        self.special_tokens = data['special_tokens']
        self.byte_seq_to_id = {k.encode('utf-8'): v for k, v in data['byte_seq_to_id'].items()}
        self.id_to_byte_seq = {int(k): v.encode('utf-8') for k, v in data['id_to_byte_seq'].items()}
        self.vocab_version = data['vocab_version']
        self.next_id = len(self.id_to_byte_seq)


class EncodingResult:
    def __init__(self, ids: List[int]):
        self.ids = ids
    
    def __len__(self):
        return len(self.ids)

import torch
import torch.nn.functional as F
from typing import Optional, Generator, Tuple
from ..models import HyenaModel
from ..tokenizers import SemanticTokenizer


class Generator:
    def __init__(
        self,
        model: HyenaModel,
        tokenizer: SemanticTokenizer,
        device: Optional[torch.device] = None
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.7,
        top_k: Optional[int] = None,
        top_p: float = 0.9,
        seq_len: int = 1024
    ) -> str:
        encoded = self.tokenizer.encode(prompt)
        input_ids = torch.tensor(encoded.ids, dtype=torch.long).unsqueeze(0).to(self.device)
        
        if input_ids.size(1) > seq_len:
            input_ids = input_ids[:, -seq_len:]
        
        generated = input_ids
        
        with torch.no_grad():
            for _ in range(max_tokens):
                if generated.size(1) > seq_len:
                    generated = generated[:, -seq_len:]
                
                with torch.amp.autocast(device_type=self.device.type, enabled=(self.device.type == 'cuda')):
                    logits = self.model(generated)
                
                next_token_logits = logits[0, -1, :] / temperature
                
                if top_k is not None:
                    values, indices = torch.topk(next_token_logits, top_k)
                    next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                    next_token_logits.scatter_(0, indices, values)
                
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                    sorted_indices_to_remove[0] = False
                    
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)
                
                if next_token.item() == self.tokenizer.special_tokens["<STOP>"]:
                    break
        
        generated_ids = generated[0].cpu().tolist()
        return self.tokenizer.decode(generated_ids)
    
    def generate_stream(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.7,
        top_k: Optional[int] = None,
        top_p: float = 0.9,
        seq_len: int = 1024
    ) -> Generator[str, None, None]:
        encoded = self.tokenizer.encode(prompt)
        input_ids = torch.tensor(encoded.ids, dtype=torch.long).unsqueeze(0).to(self.device)
        
        if input_ids.size(1) > seq_len:
            input_ids = input_ids[:, -seq_len:]
        
        generated = input_ids
        
        with torch.no_grad():
            for _ in range(max_tokens):
                if generated.size(1) > seq_len:
                    generated = generated[:, -seq_len:]
                
                with torch.amp.autocast(device_type=self.device.type, enabled=(self.device.type == 'cuda')):
                    logits = self.model(generated)
                
                next_token_logits = logits[0, -1, :] / temperature
                
                if top_k is not None:
                    values, indices = torch.topk(next_token_logits, top_k)
                    next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                    next_token_logits.scatter_(0, indices, values)
                
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                    sorted_indices_to_remove[0] = False
                    
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)
                
                token_text = self.tokenizer.decode([next_token.item()])
                yield token_text
                
                if next_token.item() == self.tokenizer.special_tokens["<STOP>"]:
                    break

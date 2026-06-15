from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    vocab_size: int = 8000
    d_model: int = 256
    n_layers: int = 6
    dim_feedforward: Optional[int] = None
    dropout: float = 0.1
    max_seq_len: int = 1024
    desired_receptive_field: int = 2048
    use_attention: bool = False
    n_heads: int = 4
    attn_dropout: float = 0.1
    prefer_reflect: bool = False
    gradient_checkpointing: bool = False
    
    def __post_init__(self):
        if self.dim_feedforward is None:
            self.dim_feedforward = self.d_model * 4
        
        if self.n_heads == 0 or self.n_heads is None:
            self.n_heads = 1
            for h in range(min(8, self.d_model), 0, -1):
                if self.d_model % h == 0:
                    self.n_heads = h
                    break

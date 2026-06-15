import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .config import ModelConfig


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1, causal: bool = True):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.causal = causal
        
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.n_heads, self.d_k)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if self.causal:
            mask = torch.triu(torch.ones(L, L, device=x.device), diagonal=1).bool()
            scores = scores.masked_fill(mask, float('-inf'))
        
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(B, L, D)
        out = self.out_proj(out)
        
        return out


class HyenaLayer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        self.ln1 = nn.LayerNorm(config.d_model) if config.use_attention else nn.Identity()
        self.attn = MultiHeadAttention(
            config.d_model, 
            config.n_heads, 
            config.attn_dropout, 
            causal=True
        ) if config.use_attention else nn.Identity()
        
        kernel_size, dilation = self._compute_kernel_params()
        self.conv = nn.Conv1d(
            in_channels=config.d_model,
            out_channels=config.d_model,
            kernel_size=kernel_size,
            stride=1,
            padding='same',
            dilation=dilation,
            groups=config.d_model,
            bias=True,
            padding_mode=config.padding_mode
        )
        
        self.gate = nn.Linear(config.d_model, config.d_model)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.dim_feedforward),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.dim_feedforward, config.d_model),
            nn.Dropout(config.dropout)
        )
    
    def _compute_kernel_params(self):
        max_k = max(3, min(self.config.max_seq_len, self.config.desired_receptive_field))
        if max_k % 2 == 0:
            max_k -= 1
        max_k = max(3, max_k)
        
        dilation = max(1, (self.config.desired_receptive_field - 1) // max(1, max_k - 1))
        kernel_size = (self.config.desired_receptive_field - 1) // max(1, dilation) + 1
        
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        kernel_size = max(3, min(int(kernel_size), int(self.config.max_seq_len)))
        if kernel_size % 2 == 0:
            kernel_size = kernel_size - 1 if kernel_size > 1 else 3
        
        return int(kernel_size), int(dilation)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.config.use_attention:
            attn_out = self.attn(self.ln1(x))
            x = x + attn_out
        
        g = torch.sigmoid(self.gate(x))
        conv_in = (x * g).transpose(1, 2)
        conv_out = self.conv(conv_in)
        conv_out = conv_out.transpose(1, 2)
        x = x + conv_out
        
        ffn_out = self.ffn(self.ln2(x))
        x = x + ffn_out
        
        return x


class HyenaModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        
        kernel_size, dilation = self._compute_kernel_params()
        self.kernel_size = kernel_size
        self.dilation = dilation
        
        self.layers = nn.ModuleList([HyenaLayer(config) for _ in range(config.n_layers)])
        self.final_ln = nn.LayerNorm(config.d_model)
        self.output = nn.Linear(config.d_model, config.vocab_size)
        
        self.ewc_enabled = False
        self.old_params = None
        self.fisher_diagonal = None
    
    def _compute_kernel_params(self):
        max_k = max(3, min(self.config.max_seq_len, self.config.desired_receptive_field))
        if max_k % 2 == 0:
            max_k -= 1
        max_k = max(3, max_k)
        
        dilation = max(1, (self.config.desired_receptive_field - 1) // max(1, max_k - 1))
        kernel_size = (self.config.desired_receptive_field - 1) // max(1, dilation) + 1
        
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        kernel_size = max(3, min(int(kernel_size), int(self.config.max_seq_len)))
        if kernel_size % 2 == 0:
            kernel_size = kernel_size - 1 if kernel_size > 1 else 3
        
        return int(kernel_size), int(dilation)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype != torch.long:
            x = x.long()
        
        x = self.embedding(x)
        
        for layer in self.layers:
            x = layer(x)
        
        x = self.final_ln(x)
        logits = self.output(x)
        
        return logits
    
    def enable_ewc(self):
        self.ewc_enabled = True
    
    def disable_ewc(self):
        self.ewc_enabled = False
    
    def compute_fisher(self, dataset, device, samples=1000, batch_size=16):
        self.to(device)
        self.eval()
        
        fisher = {n: torch.zeros_like(p, device=device) for n, p in self.named_parameters()}
        
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, drop_last=True)
        
        total_processed = 0
        for seq in loader:
            if isinstance(seq, (tuple, list)):
                seq = seq[0]
            
            seq = seq.to(device)
            self.zero_grad()
            
            with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                output = self(seq)
                loss = F.cross_entropy(
                    output[:, :-1].contiguous().view(-1, output.size(-1)),
                    seq[:, 1:].contiguous().view(-1)
                )
            
            loss.backward()
            
            for n, p in self.named_parameters():
                if p.grad is not None:
                    fisher[n] += (p.grad.detach() ** 2) / samples
            
            total_processed += seq.size(0)
            if total_processed >= samples:
                break
        
        self.fisher_diagonal = fisher
        self.old_params = {n: p.detach().clone() for n, p in self.named_parameters()}
    
    def ewc_loss(self, lamda=15.0):
        if not self.ewc_enabled or self.fisher_diagonal is None or self.old_params is None:
            return 0.0
        
        loss = 0.0
        for n, p in self.named_parameters():
            if n in self.fisher_diagonal:
                loss += (self.fisher_diagonal[n] * (p - self.old_params[n]) ** 2).sum()
        return lamda * loss

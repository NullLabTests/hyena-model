import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any
from ..models import HyenaModel, ModelConfig
from .dataset import TextDataset


class Trainer:
    def __init__(
        self,
        model: HyenaModel,
        config: Optional[Dict[str, Any]] = None
    ):
        self.model = model
        self.config = config or {}
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        self.learning_rate = self.config.get('learning_rate', 1e-4)
        self.batch_size = self.config.get('batch_size', 4)
        self.epochs = self.config.get('epochs', 10)
        self.ewc_lambda = self.config.get('ewc_lambda', 0.0)
        self.gradient_accumulation_steps = self.config.get('gradient_accumulation_steps', 1)
        self.max_grad_norm = self.config.get('max_grad_norm', 1.0)
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=0.01
        )
        
        self.scaler = torch.amp.GradScaler(enabled=(self.device.type == 'cuda'))
    
    def train(
        self,
        train_dataset: TextDataset,
        val_dataset: Optional[TextDataset] = None,
        save_path: Optional[str] = None
    ):
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True
        )
        
        if val_dataset:
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False
            )
        else:
            val_loader = None
        
        self.model.train()
        
        for epoch in range(self.epochs):
            total_loss = 0
            self.optimizer.zero_grad()
            
            for step, (input_ids, target_ids) in enumerate(train_loader):
                input_ids = input_ids.to(self.device)
                target_ids = target_ids.to(self.device)
                
                with torch.amp.autocast(device_type=self.device.type, enabled=(self.device.type == 'cuda')):
                    logits = self.model(input_ids)
                    loss = F.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        target_ids.view(-1)
                    )
                    
                    if self.ewc_lambda > 0:
                        loss += self.model.ewc_loss(self.ewc_lambda)
                
                loss = loss / self.gradient_accumulation_steps
                self.scaler.scale(loss).backward()
                
                if (step + 1) % self.gradient_accumulation_steps == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                
                total_loss += loss.item() * self.gradient_accumulation_steps
                
                if step % 100 == 0:
                    print(f"Epoch {epoch+1}/{self.epochs}, Step {step}, Loss: {loss.item():.4f}")
            
            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch+1}/{self.epochs}, Avg Loss: {avg_loss:.4f}")
            
            if val_loader:
                val_loss = self.evaluate(val_loader)
                print(f"Validation Loss: {val_loss:.4f}")
            
            if save_path:
                self.save(save_path)
        
        return self.model
    
    def evaluate(self, val_loader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for input_ids, target_ids in val_loader:
                input_ids = input_ids.to(self.device)
                target_ids = target_ids.to(self.device)
                
                with torch.amp.autocast(device_type=self.device.type, enabled=(self.device.type == 'cuda')):
                    logits = self.model(input_ids)
                    loss = F.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        target_ids.view(-1)
                    )
                
                total_loss += loss.item()
        
        self.model.train()
        return total_loss / len(val_loader)
    
    def save(self, filepath: str):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.model.config
        }, filepath)
    
    def load(self, filepath: str):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

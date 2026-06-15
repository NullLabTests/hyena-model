from .dataset import TextDataset

__all__ = ['TextDataset']

def get_trainer():
    from .trainer import Trainer
    return Trainer

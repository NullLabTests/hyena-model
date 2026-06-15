from .config import ModelConfig

__all__ = ['ModelConfig']

def get_model_classes():
    from .hyena_model import HyenaModel, MultiHeadAttention
    return HyenaModel, MultiHeadAttention

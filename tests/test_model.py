import torch
from models import HyenaModel, ModelConfig, MultiHeadAttention


def test_model_config():
    config = ModelConfig()
    assert config.d_model == 256
    assert config.n_layers == 6
    assert config.vocab_size == 8000
    print("✓ Model config test passed")


def test_multi_head_attention():
    attn = MultiHeadAttention(d_model=256, n_heads=4)
    x = torch.randn(2, 10, 256)
    out = attn(x)
    assert out.shape == (2, 10, 256)
    print("✓ Multi-head attention test passed")


def test_hyena_model():
    config = ModelConfig(d_model=128, n_layers=2, vocab_size=1000)
    model = HyenaModel(config)
    
    x = torch.randint(0, 1000, (2, 32))
    logits = model(x)
    
    assert logits.shape == (2, 32, 1000)
    print("✓ Hyena model test passed")


def test_hyena_model_with_attention():
    config = ModelConfig(d_model=128, n_layers=2, vocab_size=1000, use_attention=True)
    model = HyenaModel(config)
    
    x = torch.randint(0, 1000, (2, 32))
    logits = model(x)
    
    assert logits.shape == (2, 32, 1000)
    print("✓ Hyena model with attention test passed")


def test_ewc():
    config = ModelConfig(d_model=64, n_layers=1, vocab_size=100)
    model = HyenaModel(config)
    
    model.enable_ewc()
    assert model.ewc_enabled == True
    
    ewc_loss = model.ewc_loss(lamda=10.0)
    assert ewc_loss == 0.0
    
    print("✓ EWC test passed")


if __name__ == "__main__":
    test_model_config()
    test_multi_head_attention()
    test_hyena_model()
    test_hyena_model_with_attention()
    test_ewc()
    print("\nAll model tests passed!")

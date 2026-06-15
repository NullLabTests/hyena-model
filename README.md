<div align="center">

# 🐊 Hyena Model

**A minimal, modular implementation of the Hyena Hierarchy architecture for language modeling**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

## 🎯 What It Can Be Used For

This Hyena model is designed for **efficient language modeling** on **resource-constrained environments**. It's perfect for:

- **Research & Experimentation**: Study long-sequence modeling without massive compute
- **Edge Deployment**: Run language models on devices with limited GPU memory
- **Prototyping**: Quickly iterate on language model architectures
- **Educational Projects**: Learn about modern transformer alternatives
- **Continual Learning**: Experiment with EWC for sequential task learning
- **API Development**: Build custom language generation services

## ✨ Features

- **🚀 Hyena Hierarchy Architecture**: Efficient long-sequence modeling with depthwise convolutions
- **🎨 Optional Attention**: Hybrid mode with multi-head self-attention for local refinement
- **🧠 Elastic Weight Consolidation (EWC)**: Continual learning support
- **💾 Memory Efficient**: Optimized for smaller GPU setups (free tier compatible)
- **🧩 Modular Design**: Clean separation of concerns
- **🔤 Simple Tokenizer**: Byte-level semantic tokenizer
- **🌐 OpenAI-Compatible API**: Flask server with standard endpoints

## 📦 Installation

```bash
pip install -r requirements.txt
```

## 🚀 Quick Start

### Training

```bash
python train.py \
    --data training_data.txt \
    --vocab_size 8000 \
    --d_model 256 \
    --n_layers 6 \
    --batch_size 4 \
    --epochs 10 \
    --output model.pth \
    --tokenizer_output tokenizer.json
```

### Inference

```bash
python inference.py \
    --model model.pth \
    --tokenizer tokenizer.json \
    --prompt "Once upon a time" \
    --max_tokens 100
```

### API Server

```python
from api import create_app

app = create_app('model.pth', 'tokenizer.json')
app.run(host='0.0.0.0', port=5000)
```

## 🏗️ Architecture

### Model Configuration

```python
from models import ModelConfig

config = ModelConfig(
    vocab_size=8000,
    d_model=256,
    n_layers=6,
    dim_feedforward=1024,
    dropout=0.1,
    max_seq_len=1024,
    desired_receptive_field=2048,
    use_attention=False,
    n_heads=4
)
```

### Memory Optimization

- **Gradient Accumulation**: Effective larger batch sizes without memory overhead
- **Mixed Precision Training (AMP)**: Faster training with reduced memory usage
- **Gradient Clipping**: Stable training with controlled gradients
- **Configurable Sequence Length**: Adjust based on available memory
- **Depthwise Convolutions**: Efficient parameter usage

## 🧪 Testing

```bash
# Run model tests
python tests/test_model.py

# Run tokenizer tests
python tests/test_tokenizer.py
```

## 🔌 API Endpoints

### Completions

```bash
curl -X POST http://localhost:5000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{"prompt": "Hello world", "max_tokens": 50}'
```

### Chat Completions

```bash
curl -X POST http://localhost:5000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"messages": [{"role": "user", "content": "Hello"}]}'
```

## 📁 Module Structure

```
hyena_model/
├── models/           # Model architecture
├── tokenizers/       # Tokenizer implementation
├── training/         # Training logic
├── inference/        # Generation logic
├── api/             # Flask API server
├── utils/           # Utilities
└── tests/           # Test suite
```

## 💻 GPU Requirements

| VRAM | Configuration |
|------|---------------|
| **4GB** (Minimum) | batch_size=1, d_model=128, n_layers=4, seq_len=512 |
| **8GB** (Recommended) | batch_size=4, d_model=256, n_layers=6, seq_len=1024 |
| **16GB+** | batch_size=8, d_model=512, n_layers=12, seq_len=2048 |

**For smaller GPUs, reduce:**
- `batch_size` to 1 or 2
- `d_model` to 128 or 192
- `n_layers` to 4
- `seq_len` to 512

## 📄 License

MIT

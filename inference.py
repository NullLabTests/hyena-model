import argparse
from models import ModelConfig, get_model_classes
from tokenizers import SemanticTokenizer
from inference import get_generator
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Model checkpoint path')
    parser.add_argument('--tokenizer', type=str, required=True, help='Tokenizer path')
    parser.add_argument('--prompt', type=str, required=True, help='Input prompt')
    parser.add_argument('--max_tokens', type=int, default=100, help='Max tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.7, help='Sampling temperature')
    parser.add_argument('--top_p', type=float, default=0.9, help='Top-p sampling')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("Loading tokenizer...")
    tokenizer = SemanticTokenizer()
    tokenizer.load(args.tokenizer)
    
    print("Loading model...")
    checkpoint = torch.load(args.model, map_location=device)
    config = checkpoint['config']
    HyenaModel, _ = get_model_classes()
    model = HyenaModel(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print("Generating...")
    Generator = get_generator()
    generator = Generator(model, tokenizer, device)
    
    output = generator.generate(
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p
    )
    
    print("\nGenerated text:")
    print(output)


if __name__ == "__main__":
    main()

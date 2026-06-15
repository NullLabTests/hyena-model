import argparse
from models import ModelConfig, get_model_classes
from tokenizers import SemanticTokenizer
from training import TextDataset, get_trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help='Training data file')
    parser.add_argument('--vocab_size', type=int, default=8000, help='Vocabulary size')
    parser.add_argument('--d_model', type=int, default=256, help='Model dimension')
    parser.add_argument('--n_layers', type=int, default=6, help='Number of layers')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--seq_len', type=int, default=1024, help='Sequence length')
    parser.add_argument('--output', type=str, default='model.pth', help='Output model path')
    parser.add_argument('--tokenizer_output', type=str, default='tokenizer.json', help='Output tokenizer path')
    args = parser.parse_args()
    
    print("Training tokenizer...")
    tokenizer = SemanticTokenizer(vocab_size=args.vocab_size)
    tokenizer.train(args.data)
    tokenizer.save(args.tokenizer_output)
    print(f"Tokenizer saved to {args.tokenizer_output}")
    
    print("Creating datasets...")
    train_dataset = TextDataset(args.data, tokenizer, seq_len=args.seq_len)
    
    print("Creating model...")
    config = ModelConfig(
        vocab_size=tokenizer.get_vocab_size(),
        d_model=args.d_model,
        n_layers=args.n_layers,
        max_seq_len=args.seq_len
    )
    HyenaModel, _ = get_model_classes()
    model = HyenaModel(config)
    
    print("Starting training...")
    Trainer = get_trainer()
    trainer = Trainer(model, config={
        'learning_rate': args.learning_rate,
        'batch_size': args.batch_size,
        'epochs': args.epochs
    })
    
    trainer.train(train_dataset, save_path=args.output)
    print(f"Model saved to {args.output}")


if __name__ == "__main__":
    main()

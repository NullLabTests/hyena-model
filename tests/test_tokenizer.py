import tempfile
import os
from tokenizers import SemanticTokenizer


def test_tokenizer_init():
    tokenizer = SemanticTokenizer(vocab_size=1000)
    assert tokenizer.vocab_size == 1000
    assert tokenizer.special_tokens["<PAD>"] == 0
    assert tokenizer.special_tokens["<UNK>"] == 1
    print("✓ Tokenizer init test passed")


def test_tokenizer_train():
    tokenizer = SemanticTokenizer(vocab_size=100)
    
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        f.write("hello world this is a test hello world")
        temp_path = f.name
    
    try:
        tokenizer.train(temp_path)
        assert tokenizer.get_vocab_size() > 0
        print("✓ Tokenizer train test passed")
    finally:
        os.unlink(temp_path)


def test_tokenizer_encode_decode():
    tokenizer = SemanticTokenizer(vocab_size=100)
    
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        f.write("hello world test")
        temp_path = f.name
    
    try:
        tokenizer.train(temp_path)
        
        text = "hello world"
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded.ids)
        
        assert len(encoded.ids) > 0
        print("✓ Tokenizer encode/decode test passed")
    finally:
        os.unlink(temp_path)


def test_tokenizer_save_load():
    tokenizer = SemanticTokenizer(vocab_size=100)
    
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        f.write("hello world test")
        temp_text_path = f.name
    
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
        temp_tokenizer_path = f.name
    
    try:
        tokenizer.train(temp_text_path)
        tokenizer.save(temp_tokenizer_path)
        
        new_tokenizer = SemanticTokenizer()
        new_tokenizer.load(temp_tokenizer_path)
        
        assert new_tokenizer.get_vocab_size() == tokenizer.get_vocab_size()
        print("✓ Tokenizer save/load test passed")
    finally:
        os.unlink(temp_text_path)
        os.unlink(temp_tokenizer_path)


if __name__ == "__main__":
    test_tokenizer_init()
    test_tokenizer_train()
    test_tokenizer_encode_decode()
    test_tokenizer_save_load()
    print("\nAll tokenizer tests passed!")

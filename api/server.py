from flask import Flask, request, jsonify
from flask_cors import CORS
from ..models import HyenaModel, ModelConfig
from ..tokenizers import SemanticTokenizer
from ..inference import Generator
import torch


def create_app(model_path: str, tokenizer_path: str):
    app = Flask(__name__)
    CORS(app)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    config = ModelConfig()
    model = HyenaModel(config)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    tokenizer = SemanticTokenizer()
    tokenizer.load(tokenizer_path)
    
    generator = Generator(model, tokenizer, device)
    
    @app.route('/v1/completions', methods=['POST'])
    def completions():
        data = request.get_json()
        prompt = data.get('prompt', '')
        max_tokens = data.get('max_tokens', 100)
        temperature = data.get('temperature', 0.7)
        top_p = data.get('top_p', 0.9)
        
        generated = generator.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p
        )
        
        return jsonify({
            'text': generated,
            'usage': {
                'prompt_tokens': len(tokenizer.encode(prompt).ids),
                'completion_tokens': len(tokenizer.encode(generated).ids)
            }
        })
    
    @app.route('/v1/chat/completions', methods=['POST'])
    def chat_completions():
        data = request.get_json()
        messages = data.get('messages', [])
        max_tokens = data.get('max_tokens', 100)
        temperature = data.get('temperature', 0.7)
        
        prompt = '\n'.join([f"{m['role']}: {m['content']}" for m in messages])
        
        generated = generator.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        return jsonify({
            'choices': [{
                'message': {
                    'role': 'assistant',
                    'content': generated
                }
            }]
        })
    
    @app.route('/health', methods=['GET'])
    def health():
        return jsonify({'status': 'ok'})
    
    return app

# chatbot/tasks.py - Orchestrates chatbot training and testing.
import torch
import os
from common.config import ChatbotConfig
from common.trainer import train_model
from .data import prepare_chatbot_loaders
from transformers import GPT2TokenizerFast

def run_training(args):
    config = ChatbotConfig(args)
    train_loader, val_loader, _ = prepare_chatbot_loaders(config)
    model, criterion = config.get_model_and_criterion()
    train_model(config, model, criterion, train_loader, val_loader)

def _load_model_for_inference(config):
    tokenizer_dir = os.path.join(os.path.dirname(config.ckpt_path), "chatbot_tokenizer")
    if not os.path.exists(config.ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at {config.ckpt_path}. Please train a model first.")
    
    tokenizer = GPT2TokenizerFast.from_pretrained(tokenizer_dir)
    config.vocab_size = len(tokenizer)
    config.pad_idx = tokenizer.pad_token_id
    
    model, _ = config.get_model_and_criterion()
    model.eos_token_id = tokenizer.eos_token_id
    
    checkpoint = torch.load(config.ckpt_path, map_location=config.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(config.device).eval()
    print("--- Model and Tokenizer loaded for inference. ---")
    return model, tokenizer

def run_testing(args):
    config = ChatbotConfig(args)
    model, tokenizer = _load_model_for_inference(config)
    prompts = ["The old man sat", "In a shocking turn of events,", "The best way to learn programming is"]
    
    for prompt in prompts:
        print(f"\nPrompt: '{prompt}'")
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(config.device)
        output_ids = model.generate(input_ids, max_new_tokens=30)
        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        print(f" -> Generated: '{generated_text}'")

def run_chat(args):
    config = ChatbotConfig(args)
    model, tokenizer = _load_model_for_inference(config)
    print("\n--- Chatbot is ready. Type 'exit' to end. ---")
    
    while True:
        try:
            prompt = input("You: ")
            if prompt.lower() in ['exit', 'quit']:
                break
            input_ids = tokenizer.encode(prompt, return_tensors='pt').to(config.device)
            output_ids = model.generate(input_ids, max_new_tokens=60, temperature=0.7, top_k=40)
            response_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            clean_response = response_text.replace(prompt, '', 1).strip()
            print(f"Bot: {clean_response}")
        except KeyboardInterrupt:
            print("\nExiting chat session.")
            break
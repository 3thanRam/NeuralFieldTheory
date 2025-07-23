# tasks/chatbot_tasks.py
import torch
import os
from types import SimpleNamespace




def _run_chatbot_test(config, model, tokenizer):
    """ Runs a predefined set of prompts for testing. """
    prompts = ["The old man sat", "In a shocking turn of events,", "The best way to learn programming is"]
    for prompt in prompts:
        print(f"\nPrompt: '{prompt}'")
        generated_text = model.generate(prompt, max_new_tokens=30, tokenizer=tokenizer)
        print(f" -> Generated: '{generated_text}'")

def _run_chatbot_chat(config, model, tokenizer):
    """ Starts an interactive chat session. """
    print("\n--- Chatbot is ready. Type 'exit' to end. ---")
    while True:
        try:
            prompt = input("You: ")
            if prompt.lower() in ['exit', 'quit']: break
            generated_text = model.generate(prompt, max_new_tokens=60, tokenizer=tokenizer, temperature=0.7, top_k=40)
            clean_response = generated_text.replace(prompt, '', 1).strip()
            print(f"Bot: {clean_response}")
        except KeyboardInterrupt:
            print("\nExiting chat."); break
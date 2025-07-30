import torch
import torch.nn as nn
from tqdm import tqdm
import math 
from LNFT import LNFT


def load_model_for_inference(checkpoint_path):
    """
    Loads a model and its configuration from a checkpoint for inference.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Loading model from {checkpoint_path} for inference on {device} ---")

    # Load the checkpoint dictionary
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract the configuration that the model was trained with
    config = checkpoint['config']
    
    # Re-create the model architecture using the saved config
    model = LNFT(config["embed_dim"],config["d_hidden_dim"],config["num_blocks"],config).to(device)
    
    # Load the trained weights into the model
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Set the model to evaluation mode
    model.eval()
    
    print("Model loaded successfully.")
    return model, config


def test_model(model, config, test_loader):
    """
    Evaluates the final model on the test dataset and reports perplexity.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Evaluating model on the test set ---")
    
    model.to(device)
    model.eval()
    
    # Use the same loss function, but we only care about the main task loss
    main_criterion = nn.CrossEntropyLoss(ignore_index=config['pad_idx'])
    total_test_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            inputs = batch['input_ids'].to(device)
            targets = batch['labels'].to(device)
            
            logits = model(inputs, return_internals=False)
            
            # Calculate the cross-entropy loss
            loss = main_criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
            total_test_loss += loss.item()
            
    # Calculate average loss and perplexity
    avg_test_loss = total_test_loss / len(test_loader)
    perplexity = math.exp(avg_test_loss)
    
    print("\n--- Test Results ---")
    print(f"Average Test Loss: {avg_test_loss:.4f}")
    print(f"Perplexity: {perplexity:.4f}")
    print("--------------------")
    return avg_test_loss, perplexity

def manual_test_chat(model:LNFT,tokenizer, config,test_phrases=["He is ","The best way to learn is","My name is","Economy is the"]):
    print(f"--- Evaluating model on the test set ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    with torch.no_grad():
        for testphrase in test_phrases:
            print(f">>> Input: '{testphrase}'")
            
            # Now, pass the tokenizer into the generate method
            generated_text = model.generate(testphrase, 20, tokenizer=tokenizer)
            
            # The generate method now returns a string, so we can print it nicely
            print(f"<<< Output: {testphrase}{generated_text}")
            print("-" * 20)
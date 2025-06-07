#cli.py
import re
import torch # Added for torch.no_grad() if generate method doesn't handle it
from config import myconfig # Import the config class for type hinting

# def test_model(model,config:dict,test_prompts:list[str]): # Old signature
def test_model(model_obj: torch.nn.Module, config_obj: myconfig, test_prompts:list[str]): # New signature
    print("\n--- Testing Model Generation ---")
    # gen_args are now directly passed to model_obj.generate
    
    # The 'generation_order' and 'field_model' concepts were likely for a different model.
    # The new MFI model uses 'max_order' internally during setup.
    # For generation, it might use 'mfi_temperature_override' or 'mfi_sampling_mode'.
    # We'll use the defaults in the new generate method for now.
    # If you need to control these, pass them explicitly to generate.

    for prompt_txt in test_prompts:
        generated_txt = model_obj.generate(
            tokenizer=config_obj.tokenizer,
            start_text=prompt_txt,
            max_new_tokens=30,
            temperature=0.7, # Controls diversity of final token selection
            top_k=5,         # Controls diversity of final token selection
            # mfi_temperature_override=None, # Example: MFI specific param
            # mfi_sampling_mode="expectation", # Example: MFI specific param
            device=config_obj.device
        )
        print(f"Prompt: '{prompt_txt}' -> Generated: '{repr(generated_txt)}'\n")

def clean_response(s:str):
    # ... (keep existing clean_response function)
    s = re.sub(r' {2,}', ' ', s)
    s = re.sub(r' +\n +', '\n', s) 
    s = "\n".join([line.strip() for line in s.split('\n')])
    s = s.strip()
    return s


# def chat(model,config:dict): # Old signature
def chat(model_obj: torch.nn.Module, config_obj: myconfig): # New signature
    print("\n--- Chatting with the Model (type '\\bye' to exit) ---")
    # Similar to test_model, MFI specific generation params can be added if needed.

    # full_history_for_prompt = "" # Not used in current simple chat

    while True:
        prompt_txt = input(f"User: ")
        if "\\bye" == prompt_txt.lower():
            break
        if not prompt_txt.strip():
            continue

        current_prompt = prompt_txt # Simple, no history yet

        response_raw = model_obj.generate(
            tokenizer=config_obj.tokenizer,
            start_text=current_prompt,
            max_new_tokens=50,
            temperature=0.75,
            top_k=10,
            device=config_obj.device
        )
        response_clean = clean_response(response_raw)
        
        print(f"Bot: {response_clean}")
        
    print("Bot: Bye!")
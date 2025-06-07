#cli.py
import re
import torch.nn as nn
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
    s = re.sub(r' {2,}', ' ', s)
    s = re.sub(r' +\n +', '\n', s) 
    s = "\n".join([line.strip() for line in s.split('\n')])
    s = s.strip()
    return s


# def chat(model,config:dict): # Old signature
def chat(model_obj: nn.Module, config_obj: myconfig):
    print("\n--- Chat Mode (type '/exit' to quit) ---")
    history = []
    
    while True:
        try:
            user_input = input("You: ").strip()
            if user_input.lower() in ['/exit', '/quit']:
                break
                
            # Add conversation history
            full_prompt = "\n".join(history[-config_obj.history_len*2:] + [f"You: {user_input}", "Bot:"])
            
            response = model_obj.generate(
                tokenizer=config_obj.tokenizer,
                start_text=full_prompt,
                max_new_tokens=100,
                temperature=0.8,
                top_k=40,
                repetition_penalty=1.2,
                device=config_obj.device
            )
            
            # Extract only the new response
            new_response = response[len(full_prompt):].split('You:')[0].strip()
            clean_response = clean_response(new_response)
            
            print(f"Bot: {clean_response}")
            history.extend([f"You: {user_input}", f"Bot: {clean_response}"])
            
        except KeyboardInterrupt:
            print("\nExiting chat...")
            break
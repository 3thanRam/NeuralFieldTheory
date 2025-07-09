# cli.py
import re
import torch

def test_model(my_config, test_prompts:list[str]):
    print("\n--- Testing Model Generation ---")

    for prompt_txt in test_prompts:
        print(f"Prompt: '{prompt_txt}'")
        generated_txt = my_config.model.generate(
            tokenizer=my_config.tokenizer,
            start_text=prompt_txt,
            max_new_tokens=30,
            device=my_config.device
        )
        print(f" -> Generated: '{repr(generated_txt)}'\n")

def clean_response(s:str):
    s = re.sub(r' {2,}', ' ', s)
    s = re.sub(r' +\n +', '\n', s)
    s = "\n".join([line.strip() for line in s.split('\n')])
    s = s.strip()
    return s

def chat(my_config):
    print("\n--- Chat Mode (type '/exit' to quit) ---")
    history = []
    # history_len can be a new hyperparameter in config if you want
    history_len = 5

    while True:
        try:
            user_input = input("You: ").strip()
            if user_input.lower() in ['/exit', '/quit']:
                break

            # Use BOS token to signal start of generation
            full_prompt = "\n".join(history[-history_len*2:] + [f"You: {user_input}", "Bot:"]) + f" {my_config.tokenizer.bos_token}"

            response = my_config.model.generate(
                tokenizer=my_config.tokenizer,
                start_text=full_prompt,
                max_new_tokens=100,
                temperature=0.8,
                top_k=40,
                device=my_config.device
            )

            # Extract only the new response
            new_response = clean_response(response[len(full_prompt):].split('You:')[0].strip())

            print(f"Bot: {new_response}")
            history.extend([f"You: {user_input}", f"Bot: {new_response}"])

        except KeyboardInterrupt:
            print("\nExiting chat...")
            break
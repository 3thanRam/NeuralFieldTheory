# main.py
from training import train_model, get_training_texts
from cli import test_model, chat
from config import system_config

def main():
    system = system_config(load=True, mode="test", model_type="hamiltonian")
    
    if system.mode == "train":
        train_texts, val_texts = get_training_texts(system)
        if not train_texts:
            print("No training data generated. Exiting.")
            return
            
        system.raw_training_texts = train_texts
        system.raw_validation_texts = val_texts
        train_model(system)
        
    elif system.mode == "test":
        if not system.load:
            print(f"Warning: Running {system.model_type} in test mode but 'load' is False. Using an uninitialized model.")
        test_model(system, [
            "The old man sat",
            "In a shocking turn of events,",
            "The best way to learn programming is",
            "Once upon a time"
        ])
        
    elif system.mode == "chat":
        if not system.load:
            print(f"Warning: Running {system.model_type} in chat mode but 'load' is False. Using an uninitialized model.")
        chat(system)

if __name__ == "__main__":
    main()
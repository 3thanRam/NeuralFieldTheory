# main.py
from training import train_model, get_training_texts
from cli import test_model, chat # Added chat import if you intend to use it
from config import myconfig


def main():
    # Consider adding argparse here for command-line arguments like mode, load, etc.
    # For now, using default config values or manually changing them in myconfig.
    system = myconfig(load=True, mode="train") # Example: train mode, no load
    # system = myconfig(load=True, mode="test") # Example: test mode, with load

    if system.mode == "train":
        # Populate raw_training_texts and raw_validation_texts in the config object
        train_texts, val_texts = get_training_texts(system)
        if not train_texts:
            print("No training data generated. Exiting.")
            return
            
        system.raw_training_texts = train_texts
        system.raw_validation_texts = val_texts
        train_model(system)
    elif system.mode == "test":
        if not system.load: # Typically, testing implies loading a trained model
            print("Warning: Running in test mode but 'load' is False. Model will be uninitialized.")
        # model.eval() is called inside the generate method
        test_model(system.model, system, ["I am","five times seven is ","He died in","How are you?"])
    elif system.mode == "chat": # Example for chat mode
        if not system.load:
             print("Warning: Running in chat mode but 'load' is False. Model will be uninitialized.")
        chat(system.model, system)


if __name__ == "__main__":
    main()
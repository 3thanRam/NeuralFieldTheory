
from tasks.chatbot_tasks import _run_chatbot_test,_run_chatbot_chat
from tasks.stockbot_tasks import _run_stockbot_test
def test_model(config, model,val_loader,tokenizer):

    print(f"testmode:{config.test_type}")
    if config.test_type=="completetext":
        _run_chatbot_test(config, model, tokenizer)
    elif config.test_type=="chatbot":
         _run_chatbot_chat(config, model, tokenizer)
    elif config.test_type=="stockbot":
        _run_stockbot_test(config, model, val_loader)
    else:
        print(f"No testing mode called: {config.test_type}")
    return 
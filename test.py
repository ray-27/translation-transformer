from inference import translate
import torch
import config
import json

def val():
    eng_vocab_path = "model_loads/eng_vocab.json"
    hin_vocab_path = "model_loads/hin_vocab.json"

    with open(eng_vocab_path, 'r') as json_file:
        eng_vocab = json.load(json_file)  # Load data from JSON file into a dictionary

    with open(hin_vocab_path, 'r') as json_file:
        hin_vocab = json.load(json_file) 
    
    with open(eng_vocab_path,'r') as json_file:
        eng_index = json.load(json_file)
        inverted_eng_dict = {value: key for key, value in eng_index.items()}
    
    model = torch.load('model_loads/translation_model.pth').to('cpu')

    seq_len = config.seq_len

    x = "How are you"

    hindi = translate(x,model,seq_len,inverted_eng_dict,hin_vocab)


    return hindi

if __name__ == "__main__":
    print(val())
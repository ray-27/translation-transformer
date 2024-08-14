import json
import pandas as pd
import torch
eng_vocab_path = "model_loads/eng_vocab.json"
hin_vocab_path = "model_loads/hin_vocab.json"
# Reading JSON data
 # Load data from JSON file into a dictionary
from dataset import translation_dataset


def tensor_to_sentence(x,vocab):
    if type(x) == torch.Tensor:
        x = x.tolist()
    
    for i in x:
        word = vocab[f'{i}']
        if word == "<start>":
            continue
        elif word == "<end>":
            break
        else:
            print(word)

if __name__ == "__main__":
    with open(eng_vocab_path, 'r') as json_file:
        eng_vocab = json.load(json_file)  # Load data from JSON file into a dictionary

    with open(hin_vocab_path, 'r') as json_file:
        hin_vocab = json.load(json_file) 

    data = pd.read_csv('data/hugging_data.csv')
    dataset = translation_dataset(data)
    for e,h in dataset:
        break

    x = h
    tensor_to_sentence(x,hin_vocab)